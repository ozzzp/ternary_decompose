import torch
from itertools import count
from einops import rearrange

def _ternary_policy(a, cos_thresh):
    if cos_thresh > 0:
        sorted_a = torch.sort(torch.abs(a), dim=-1, descending=True)[0]
        top_k_sum = torch.cumsum(sorted_a, dim=-1)
        arange = torch.arange(1, a.shape[-1] + 1, dtype=torch.float32, device=a.device)
        top_k_scalar = torch.sqrt(arange)
        score = top_k_sum/top_k_scalar
        idx_1 = torch.argmax(torch.ge(score, cos_thresh).float(), dim=-1)
        idx_2 = torch.argmax(score, dim=-1)
        is_satisfy = torch.ge(torch.gather(score, index=idx_1.unsqueeze(-1), dim=1).squeeze(-1), cos_thresh)
        idx = torch.where(is_satisfy, idx_1, idx_2)
        thresh = torch.gather(sorted_a, index=idx.unsqueeze(-1), dim=1).squeeze(-1)

        #thresh = jnp.sort(jnp.abs(a.reshape([-1])))[int(sparsity_rate * a.size)]
        out = torch.where(a>=0, torch.ones_like(a), -torch.ones_like(a))
        return torch.where(torch.abs(a) < thresh.unsqueeze(-1), torch.zeros_like(a), out)
    else:
        out = torch.where(a >= 0, torch.ones_like(a), -torch.ones_like(a))
        return out

def _find_primary_ternary_component(A, cos_thresh, top_k):
    # find binary approximation of the top k component of A
    if A.shape[-2:].numel() > 1024 ** 2:
        u, s, v = torch.svd_lowrank(A, 5 * top_k)
        v = v.transpose(-1, -2)
    else:
        u, s, v = torch.linalg.svd(A, full_matrices=False)
    u = u[..., :, 0:top_k]
    u = _ternary_policy(rearrange(u, "b l k -> (b k) l"), cos_thresh)
    u = rearrange(u, "(b k) l -> b l k", k=top_k)

    v = v[..., 0:top_k, :]
    v = _ternary_policy(rearrange(v, "b k l -> (b k) l"), cos_thresh)
    v = rearrange(v, "(b k) l -> b k l", k=top_k)
    return u, s[:, 0], v

def _get_best_s(u, v, A, damping=1e-5):
    # find best s, such minimize \| u diag(s) v - A\|_F
    x = (A @ v.transpose(1, 2) * u).sum(dim=1)
    kernel = (u.transpose(1, 2) @ u) * (v @ v.transpose(1, 2))

    mean_s = torch.pow(kernel, 2).sum() / kernel.shape[0] / kernel.shape[1]
    mean_s = torch.pow(mean_s, 0.5)
    kernel += damping * mean_s * torch.eye(kernel.shape[-1], device=kernel.device)
    upper, _ = torch.linalg.cholesky_ex(kernel)
    x = torch.cholesky_solve(x.unsqueeze(dim=-1), upper).squeeze(-1)

    #eigval, eigvec = torch.linalg.eigh(kernel)
    #eigval = torch.where(eigval > 0, 1 / eigval, torch.zeros_like(eigval))
    #x =   (eigvec @ (eigval.unsqueeze(dim=-1) * (eigvec.transpose(1, 2) @ x.unsqueeze(dim=-1)))).squeeze(-1)
    return x, A - u @ (x.unsqueeze(dim=-1) * v)

def ternary_decomposition(A, pre_u=None, pre_v=None, max_rank=None, stride=1, tolerance=0., cos_thresh=0.5):
    assert len(A.shape) >= 2
    batch_shape = A.shape[:-2]
    A = A.reshape((-1,)+A.shape[-2:])
    if A.shape[-2] > A.shape[-1]:
        A = torch.permute(A, [0, 2, 1])
        transpose = True
    else:
        transpose = False
    if pre_u is None or pre_v is None:
        assert pre_u is None and pre_v is None
        us = None
        vs = None
        rest = A
        A_s = None
        final_s = None
        if max_rank is not None:
            plan = range(0, max_rank, stride)
        else:
            plan = count(step=stride)
    else:
        us = pre_u.reshape((-1,) + pre_u.shape[-2:])
        vs = pre_v.reshape((-1,) + pre_v.shape[-2:])

        lens = (us.shape[-1] // stride) * stride
        if lens > 0:
            mask = torch.randperm(us.shape[-1])[:lens]
            us = us[..., :, mask]
            vs = vs[..., mask, :]

            if transpose:
                us, vs = torch.permute(vs, [0, 2, 1]), torch.permute(us, [0, 2, 1])

            final_s, rest = _get_best_s(us, vs, A)
            _, A_s, _ = _find_primary_ternary_component(A, cos_thresh, 1)
        else:
            us = None
            vs = None
            rest = A
            A_s = None
            final_s = None

        if max_rank is not None:
            plan = range(lens, max_rank, stride)
        else:
            plan = count(step=stride)

    for i in plan:
        u, s, v = _find_primary_ternary_component(rest, cos_thresh, stride)
        if A_s is None:
            A_s = s
        if torch.max(s / A_s) < tolerance:
            break
        if us is None or vs is None:
            us = u
            vs = v
        else:
            us = torch.concat([us, u], dim=-1)
            vs = torch.concat([vs, v], dim=-2)

        final_s, rest = _get_best_s(us, vs, A)

    if transpose:
        us, vs = torch.permute(vs, [0, 2, 1]), torch.permute(us, [0, 2, 1])
    us = us.reshape(batch_shape + us.shape[1:])
    vs = vs.reshape(batch_shape + vs.shape[1:])
    final_s = final_s.reshape(batch_shape + final_s.shape[1:])

    return us, final_s, vs

def _get_critical_rank(A, bits, sparsity):
    M, N = A.shape[-2:]
    muls = torch.count_nonzero(torch.logical_not(torch.isin(A, torch.tensor([-1, 0, 1], dtype=A.dtype, device=A.device))))
    adds = torch.count_nonzero(torch.logical_not(torch.isin(A, torch.tensor([0], dtype=A.dtype, device=A.device))))
    return ((bits - 2) * int(muls) + int(adds)) / ((bits - 2) + (1 - sparsity) * (M + N))

def check(A, U, S, V, bits=8, never_mind_sparsity=False):
    rank = U.shape[-1]
    rest = A -  U @ (S.unsqueeze(-1) * V)
    def get_operator_norm(x):
        if A.shape[-2:].numel() > 1024 ** 2:
            _, s, _ = torch.svd_lowrank(x, 10)
        else:
            _, s, _ = torch.linalg.svd(x)
        return s[..., 0]
    error = torch.max(get_operator_norm(rest)/get_operator_norm(A))

    assert torch.isin(U, torch.tensor([-1, 0, 1], dtype=torch.float32, device=U.device)).all(), torch.unique(U)
    assert torch.isin(V, torch.tensor([-1, 0, 1], dtype=torch.float32, device=V.device)).all(), torch.unique(V)

    sparsity = (torch.count_nonzero(U==0) + torch.count_nonzero(V==0)) / (U.numel() + V.numel())
    if never_mind_sparsity:
        max_rank = _get_critical_rank(A, bits, 0)
    else:
        max_rank = _get_critical_rank(A, bits, sparsity)
    return error, rank, max_rank, sparsity

def get_s(A, U, V, cos_thresh):
    assert len(A.shape) >= 2
    batch_shape = A.shape[:-2]
    A = A.reshape((-1,) + A.shape[-2:])
    U = U.reshape((-1,) + U.shape[-2:])
    V = V.reshape((-1,) + V.shape[-2:])

    s, rest_A = _get_best_s(U, V, A)
    u, _, v = _find_primary_ternary_component(rest_A, cos_thresh, 1)

    rest_s, _ = _get_best_s(u, v, rest_A)
    rest_s *= (torch.abs(u).sum(dim=-2) * torch.abs(v).sum(dim=-1)) ** 0.5

    s = s.reshape(batch_shape + s.shape[1:])
    rest_s = rest_s.reshape(batch_shape)
    return s, rest_s

if __name__ == "__main__":
    A = torch.distributions.Gamma(1, 1).sample([10, 256, 512]).cuda()
    A = torch.copysign(A, torch.randn(A.shape, device=A.device))

    for i in torch.flip(torch.arange(0, 1, 0.05, dtype=torch.float32), dims=[0]):
        u, s, v = ternary_decomposition(A, tolerance=1e-2, cos_thresh=float(i), stride=5)
        error, rank, max_rank, sparsity = check(A, u, s, v, bits=8)
        print("{:.3g}, error: {:.3g}, rank: {}/{}, cost:{:.3g}, sparsity: {}".format(i, error, rank, max_rank,
                                                                                     rank / max_rank, sparsity))