import torch
import torch.nn.functional as F
from .base import TernSVDBase, LogItem
from .tern_svd_conv2d import Ternary_SVD_Conv2D

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

class Ternary_SVD_ConvTranspose(torch.nn.ConvTranspose2d, TernSVDBase):
    W_src_shape = Ternary_SVD_Conv2D.W_src_shape
    uv_form = Ternary_SVD_Conv2D.uv_form
    reshape_w = Ternary_SVD_Conv2D.reshape_w
    recover_w = Ternary_SVD_Conv2D.recover_w
    reshape_usv = Ternary_SVD_Conv2D.reshape_usv
    recover_usv = Ternary_SVD_Conv2D.recover_usv
    get_form_order = Ternary_SVD_Conv2D.get_form_order
    weight_to_usv = Ternary_SVD_Conv2D.weight_to_usv
    _usv_to_weight = Ternary_SVD_Conv2D._usv_to_weight

    def __init__(self, *args, **kwargs):
        super(Ternary_SVD_ConvTranspose, self).__init__(*args, **kwargs)

    def batch_mTv(self, W, X):
        K, M = W.shape
        W = W.reshape([self.groups, -1, M])
        X = X.reshape([self.groups, -1, 1])
        WX = torch.matmul(W.transpose(1, 2), X)
        return WX.reshape([-1])

    def forward(self, x, output_size = None):
        if hasattr(self, 'weight'):
            return super(Ternary_SVD_ConvTranspose, self).forward(x, output_size=output_size)
        else:
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

            x = F.conv_transpose2d(x, self.u, None, stride_u,
                                   padding=(0, 0), dilation=dilation_u, groups=self.groups)
            x *= self.s[None, :, None, None]
            x = F.conv_transpose2d(x, self.v, self.bias, stride_v,
                                   padding=(0, 0), dilation=dilation_v, groups=self.groups)

            if self.padding_mode != 'zeros':
                raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

            output_padding = self._output_padding(
                x, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
                num_spatial_dims, self.dilation)  # type: ignore[arg-type]
            padding = pair(self.padding)

            x = F.pad(x, [- padding[1], output_padding[1] - padding[1], -padding[0], output_padding[0] - padding[0]])
            return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        self.regiter_svd_buffer(state_dict, prefix)

        super(Ternary_SVD_ConvTranspose, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                                     missing_keys, unexpected_keys, error_msgs)
