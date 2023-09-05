import torch
import torch.nn.functional as F
from .base import TernSVDBase
from einops import rearrange
import math

def pair(out):
    if isinstance(out, tuple) or isinstance(out, list):
        if len(out) == 2:
            return tuple(out)
        else:
            return  (out[0], out[0])
    else:
        return (out, out)

def squeeze_form(str):
    str = [i for i in str.split(' ') if i != '1']
    return ' '.join(str)

class Ternary_SVD_Conv2D(torch.nn.Conv2d, TernSVDBase):
    W_src_shape = "(G C_out) C_in K_1 K_2"
    uv_form = (("1 1", "K_1 K_2"),
               ("K_1 K_2", "1 1"),
               ("K_1 1", "1 K_2"),
               ("1 K_2", "K_1 1"))

    def __init__(self, *args, **kwargs):
        super(Ternary_SVD_Conv2D, self).__init__(*args, **kwargs)

    def reshape_w(self, i, w):
        u_form, v_form = self.uv_form[i]
        return rearrange(w, "{} -> G (C_out {}) (C_in {})".format(self.W_src_shape,
                                                                  squeeze_form(u_form),
                                                                  squeeze_form(v_form)),
                         G=self.groups)

    def recover_w(self, i, w):
        u_form, v_form = self.uv_form[i]
        return rearrange(w, "G (C_out {}) (C_in {}) -> {}".format(squeeze_form(u_form),
                                                                  squeeze_form(v_form),
                                                                  self.W_src_shape),
                  K_1=self.weight.shape[-2], K_2=self.weight.shape[-1])

    def reshape_usv(self, i, u, s, v):
        u_form, v_form = self.uv_form[i]
        if s is not None:
            s = rearrange(s, "(G S) -> G S", G=self.groups)
        if u is not None:
            u = rearrange(u, "(G C_out) S {} -> G (C_out {}) S".format(u_form,
                                                                       squeeze_form(u_form)),
                          G=self.groups)
        if v is not None:
            v = rearrange(v, "(G S) C_in {} -> G S (C_in {})".format(v_form,
                                                                     squeeze_form(v_form)),
                          G=self.groups)

        return u, s, v

    def recover_usv(self, i, u, s, v):
        u_form, v_form = self.uv_form[i]
        if s is not None:
            s = rearrange(s, "G S -> (G S)")

        def args(form):
            _args = {}
            if 'K_1' in form:
                _args['K_1'] = self.weight.shape[-2]
            if 'K_2' in form:
                _args['K_2'] = self.weight.shape[-1]
            return _args

        if u is not None:
            u = rearrange(u, "G (C_out {}) S -> (G C_out) S {}".format(squeeze_form(u_form),
                                                                   u_form),
                        **args(u_form))
        if  v is not None:
            v = rearrange(v, "G S (C_in {}) -> (G S) C_in {}".format(squeeze_form(v_form),
                                                                     v_form),
                          **args(v_form))
        return u, s, v

    def batch_mv(self, W, X):
        M, K = W.shape
        W = W.reshape([self.groups, -1, K])
        X = X.reshape([self.groups, K, 1])
        WX = torch.matmul(W, X)
        return WX.reshape([-1])

    def forward(self, x):
        if hasattr(self, 'weight'):
            return super(Ternary_SVD_Conv2D, self).forward(x)
        else:
            if self.padding_mode != 'zeros':
                x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            else:
                padding = pair(self.padding)
                x = F.pad(x, [padding[1], padding[1], padding[0], padding[0]])

            stride = pair(self.stride)
            dilation = pair(self.dilation)

            def process(u, v, pair):
                if u.shape[-2] > 1:
                    uv_1 = (pair[0], 1)
                else:
                    uv_1 = (1, pair[0])

                if u.shape[-1] > 1:
                    uv_2 = (pair[1], 1)
                else:
                    uv_2 = (1, pair[1])

                return tuple(zip(uv_1, uv_2))


            stride_u, stride_v = process(self.u, self.v, stride)
            dilation_u, dilation_v = process(self.u, self.v, dilation)

            x = F.conv2d(x, self.v, None, stride_v,
                         (0, 0), dilation_v, self.groups)
            x *= self.s[None, :, None, None]
            x = F.conv2d(x, self.u, self.bias, stride_u,
                         (0, 0), dilation_u, self.groups)
            return x

    def get_form_order(self, u=None, v=None):
        if u is None:
            u = self.u
        if v is None:
            v = self.v

        for j, (u_form, v_form) in enumerate(self.uv_form):
            is_order = True
            u_idx_1 = [i for i, key in enumerate(u_form.split(' ')) if key == '1']
            for i in u_idx_1:
                if u.shape[i+2] != 1:
                    is_order = False
                    break
            if not is_order:
                continue
            v_idx_1 = [i for i, key in enumerate(v_form.split(' ')) if key == '1']
            for i in v_idx_1:
                if v.shape[i + 2] != 1:
                    is_order = False
                    break
            if is_order:
                break
        return j

    def weight_to_usv(self, transform_fun, s_fun, *args, prune_rate=1, allow_split=True, allow_large_kernel=True, **kwargs):
        with torch.no_grad():
            if self.weight.numel() // self.groups <= 9:
                # it is not cost-efficient
                return

            def re_tern_usv(weight, pre_u=None, pre_v=None):
                if pre_u is None and pre_v is None:
                    idx = []
                    if allow_split:
                        idx.extend([0, 1])
                    if allow_large_kernel:
                        idx.extend([2, 3])
                    result = [transform_fun(self.reshape_w(i, weight)) for i in idx]
                    cost, order, u, s, v = min([(c, i, u, s, v) for i, (c, u, s, v) in enumerate(result)])
                else:
                    order = self.get_form_order(u=pre_u, v=pre_v)
                    weight = self.reshape_w(order, weight)
                    pre_u, _, pre_v = self.reshape_usv(order, pre_u, None, pre_v)
                    _, u, s, v = transform_fun(weight, pre_u=pre_u, pre_v=pre_v)
                u, s, v = self.recover_usv(order, u, s, v)
                return u, s, v

            if not hasattr(self, 'u') or not hasattr(self, 'v') or not hasattr(self, 's'):
                assert not hasattr(self, 'u') and not hasattr(self, 'v') and not hasattr(self, 's')
                u, s, v = re_tern_usv(self.weight)

                self.register_buffer('s', s.contiguous())
                self.register_buffer('u', u.contiguous())
                self.register_buffer('v', v.contiguous())
            else:
                form_order = self.get_form_order()

                weight = self.weight + self.rest_weight
                w = self.reshape_w(form_order, weight)
                u, _, v = self.reshape_usv(form_order, self.u, self.s, self.v)
                if math.isfinite(prune_rate):
                    s, rest_s = s_fun(w, u, v)
                    _s = torch.abs(s) * (torch.abs(u).sum(dim=-2) * torch.abs(v).sum(dim=-1)) ** 0.5
                    mask = _s  >= rest_s[:, None] * prune_rate
                    if mask.all():
                        self.s.copy_(s.reshape(self.s.shape))
                        return
                    else:
                        prune_num = torch.count_nonzero(mask, dim=-1)
                        prune_num = torch.min(prune_num)
                        if prune_num > 0:
                            idx = torch.topk(torch.abs(_s), k=int(prune_num), dim=-1)[1]
                            pre_u = torch.gather(u, dim=-1, index=idx[:, None, :].expand([u.shape[0], u.shape[1], -1]))
                            pre_v = torch.gather(v, dim=-2, index=idx[:, :, None].expand([v.shape[0], -1, v.shape[2]]))
                            pre_u, _, pre_v = self.recover_usv(form_order, pre_u, None, pre_v)
                        else:
                            pre_u, pre_v = None, None
                else:
                    pre_u, pre_v = None, None

                del self.s
                del self.u
                del self.v
                u, s, v = re_tern_usv(weight, pre_u=pre_u, pre_v=pre_v)
                self.register_buffer('s', s.contiguous())
                self.register_buffer('u', u.contiguous())
                self.register_buffer('v', v.contiguous())

            rest_w = self._usv_to_weight()
            if not hasattr(self, 'rest_weight'):
                self.register_buffer('rest_weight', rest_w)
            else:
                self.rest_weight += rest_w

    def _usv_to_weight(self):
        form_order = self.get_form_order()
        u, s, v = self.reshape_usv(form_order, self.u, self.s, self.v)
        w = u @ (s[:, :, None] * v)
        w = self.recover_w(form_order, w)
        rest_w = self.weight - w
        self.weight.data.copy_(w)
        return rest_w


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        self.regiter_svd_buffer(state_dict, prefix)

        super(Ternary_SVD_Conv2D, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                              missing_keys, unexpected_keys, error_msgs)

    def extra_repr(self) -> str:
        repr = super(Ternary_SVD_Conv2D, self).extra_repr()
        if hasattr(self, 'u') or hasattr(self, 'v'):
            assert hasattr(self, 'u') and hasattr(self, 'v')
            sparsity = float((torch.count_nonzero(self.u) + torch.count_nonzero(self.v)) / (self.u.numel() + self.v.numel()))
            repr += ', form_type={}, rank={}, sparsity={}'.format(self.get_form_order(self.u, self.v), self.s.numel() / self.groups, sparsity)
        return repr
