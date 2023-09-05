import jax
import jax.numpy as jnp
from itertools import count
from functools import partial

def _ternary_policy(a, cos_thresh):
    sorted_a = jnp.sort(jnp.abs(a.reshape([-1])))[::-1]
    top_k_sum = jnp.cumsum(sorted_a)
    arange = jnp.arange(1, top_k_sum.size+1, dtype=jnp.float32)
    top_k_scalar = jnp.sqrt(arange)
    score = (top_k_sum/top_k_scalar)
    idx_1 = jnp.argmax(score >= cos_thresh)
    idx_2 = jnp.argmax(score)
    idx = jnp.where(score[idx_1] >= cos_thresh, idx_1, idx_2)
    thresh = sorted_a[idx]

    #thresh = jnp.sort(jnp.abs(a.reshape([-1])))[int(sparsity_rate * a.size)]
    out = jnp.where(a>=0, jnp.ones_like(a), -jnp.ones_like(a))
    return jnp.where(jnp.abs(a) < thresh, jnp.zeros_like(a), out)

@partial(jax.vmap, in_axes=(0, None, None))
def _find_primary_ternary_component(A, cos_thresh, top_k):
    # find binary approximation of the top k component of A
    u, s, v = jnp.linalg.svd(A, full_matrices=False)
    u = jax.vmap(_ternary_policy, in_axes=(-1, None), out_axes=-1)(u[..., :, 0:top_k], cos_thresh)
    v = jax.vmap(_ternary_policy, in_axes=(-2, None), out_axes=-2)(v[..., 0:top_k, :], cos_thresh)
    return u, s[0], v

@jax.vmap
def _get_best_s(u, v, A):
    # find best s, such minimize \| u diag(s) v - A\|_F
    x = jnp.einsum("ik,kj,ij->k", u, v, A)
    kernel = jnp.einsum("ik,kj,il,lj->kl", u, v, u, v)
    eigval, eigvec = jnp.linalg.eigh(kernel)
    eigval = jnp.where(eigval > 0, 1 / eigval, jnp.zeros_like(eigval))
    x = jnp.einsum('ik,k,kj,j->i', eigvec, eigval, eigvec.T, x)
    return x, A - jnp.einsum("ik,k,kj->ij", u, x, v)

def ternary_decomposition(A, max_rank=None, stride=1, tolerance=0., cos_thresh=0.5):
    assert len(A.shape) >= 2
    batch_shape = A.shape[:-2]
    A = A.reshape((-1,)+A.shape[-2:])
    if A.shape[-2] > A.shape[-1]:
        A = jnp.transpose(A, [0, 2, 1])
        transpose = True
    else:
        transpose = False

    us = None
    vs = None
    final_s = None
    rest = A
    A_s = None
    if max_rank is not None:
        plan = range(0, max_rank, stride)
    else:
        plan = count(step=stride)

    for i in plan:
        u, s, v = jax.jit(_find_primary_ternary_component, static_argnums=[2])(rest, cos_thresh, stride)
        if A_s is None:
            A_s = s
        #print("sparsity: {:.3g}, cost: {}/{}, error: {:.3g}".format(sparsity, int(i), int(rank), s / l2_s[0]))
        if jnp.max(s / A_s) < tolerance:
            break
        if us is None or vs is None:
            us = u
            vs = v
        else:
            us = jnp.concatenate([us, u], axis=-1)
            vs = jnp.concatenate([vs, v], axis=-2)

        final_s, rest = jax.jit(_get_best_s)(us, vs, A)

    if transpose:
        us, vs = jnp.transpose(vs, [0, 2, 1]), jnp.transpose(us, [0, 2, 1])
    us = us.reshape(batch_shape + us.shape[1:])
    vs = vs.reshape(batch_shape + vs.shape[1:])
    final_s = final_s.reshape(batch_shape + final_s.shape[1:])

    return us, final_s, vs

def _get_critical_rank(A, bits, sparsity):
    M, N = A.shape[-2:]
    B = A.reshape([-1, M, N]).shape[0]
    muls = jnp.count_nonzero(jnp.logical_not(jnp.isin(A, jnp.array([-1, 0, 1]))))
    adds = jnp.count_nonzero(jnp.logical_not(jnp.isin(A, jnp.array([0]))))
    return (bits * muls + adds) / (bits + (1 - sparsity) * (M + N)) / B

def check(A, U, S, V, bits=8, never_mind_sparsity=False):
    rank = U.shape[-1]
    rest = A - jnp.einsum("...ik,...k,...kj->...ij", U, S, V)
    def get_operator_norm(x):
        return jnp.linalg.svd(x, compute_uv=False)[..., 0]
    error = jnp.max(get_operator_norm(rest)/get_operator_norm(A))

    assert jnp.isin(U, jnp.array([-1, 0, 1])).all()
    assert jnp.isin(V, jnp.array([-1, 0, 1])).all()

    sparsity = (jnp.count_nonzero(U==0) + jnp.count_nonzero(V==0)) / (U.size + V.size)
    if never_mind_sparsity:
        max_rank = _get_critical_rank(A, bits, 0)
    else:
        max_rank = _get_critical_rank(A, bits, sparsity)
    return error, rank, max_rank, sparsity

if __name__ == "__main__":
    A = jax.random.gamma(jax.random.PRNGKey(0), 1, shape=[1, 3*3])
    A = jnp.copysign(A, jax.random.normal(jax.random.PRNGKey(0), shape=A.shape))

    #for i in jnp.arange(0, 1, 0.05)[::-1]:
    u, s, v = ternary_decomposition(A, tolerance=1e-2, cos_thresh=0.86, stride=1)
    error, rank, max_rank, sparsity = check(A, u, s, v, bits=8)
    print("{:.3g}, error: {:.3g}, rank: {}/{}, cost:{:.3g}, sparsity: {}".format(i, error, rank, max_rank,
                                                                                 rank / max_rank, sparsity))