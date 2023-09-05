import torch
import torch.nn.functional as F
from .base import TernSVDBase
import math

class Ternary_SVD_Linear(torch.nn.Linear, TernSVDBase):
    def __init__(self, *args, **kwargs):
        super(Ternary_SVD_Linear, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'weight'):
            return super(Ternary_SVD_Linear, self).forward(x)
        else:
            x = F.linear(x, self.v.to(x.dtype).to(x.device))
            x *= self.s
            x = F.linear(x, self.u.to(x.dtype).to(x.device), bias=self.bias)
            return x

    def weight_to_usv(self, transform_fun, s_fun, *args, prune_rate=1, **kwargs):
        with torch.no_grad():
            if self.weight.numel() <= 9:
                # it is not cost-efficient
                return
            if not hasattr(self, 'u') or not hasattr(self, 'v') and not hasattr(self, 's'):
                assert not hasattr(self, 'u') and not hasattr(self, 'v') and not hasattr(self, 's')
                _, u, s, v = transform_fun(self.weight)
                self.register_buffer('s', s.contiguous())
                self.register_buffer('u', u.contiguous())
                self.register_buffer('v', v.contiguous())
            else:
                W = self.weight + self.rest_weight
                if math.isfinite(prune_rate):
                    s, rest_s = s_fun(W, self.u, self.v)
                    _s = s * (torch.abs(self.u).sum(dim=-2) * torch.abs(self.v).sum(dim=-1)) ** 0.5
                    mask = _s >= rest_s * prune_rate
                    if mask.all():
                        self.s.copy_(s)
                        return
                    else:
                        prune_num = torch.count_nonzero(mask)
                        if prune_num > 0:
                            pre_u = self.u[:, mask]
                            pre_v = self.v[mask, :]
                        else:
                            pre_u, pre_v = None, None
                else:
                    pre_u, pre_v = None, None

                del self.s
                del self.u
                del self.v
                _, u, s, v = transform_fun(W, pre_u=pre_u, pre_v=pre_v)
                self.register_buffer('s', s.contiguous())
                self.register_buffer('u', u.contiguous())
                self.register_buffer('v', v.contiguous())

            rest_w = self._usv_to_weight()
            if not hasattr(self, 'rest_weight'):
                self.register_buffer('rest_weight', rest_w)
            else:
                self.rest_weight += rest_w

    def _usv_to_weight(self):
        w = self.u @ (self.s[:, None] * self.v)
        rest_w = self.weight - w
        self.weight.data.copy_(w)
        return rest_w

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        self.regiter_svd_buffer(state_dict, prefix)

        super(Ternary_SVD_Linear, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                              missing_keys, unexpected_keys, error_msgs)

    def extra_repr(self) -> str:
        repr = super(Ternary_SVD_Linear, self).extra_repr()
        if hasattr(self, 'u') or hasattr(self, 'v'):
            assert hasattr(self, 'u') and hasattr(self, 'v')
            sparsity = float((torch.count_nonzero(self.u) + torch.count_nonzero(self.v)) / (self.u.numel() + self.v.numel()))
            repr += ', rank={}, sparsity={:.3g}'.format(self.s.numel(), sparsity)
        return repr
