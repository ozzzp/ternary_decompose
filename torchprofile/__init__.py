import torch

from .profile import profile_mul_and_add
from .version import __version__

__all__ = ['profile_mul_and_add', 'count_mul_and_add_for_first_input']


def count_mul_and_add_for_first_input(model):
    _forward = model.__class__.forward
    tested = False

    def forward(self, *args, **kwargs):
        nonlocal tested
        if not tested and not torch.jit.is_tracing():
            class _M(torch.nn.Module):
                def __init__(self, model):
                    super(_M, self).__init__()
                    self.model = model
                    self.keys = tuple(kwargs.keys())

                def forward(self, *_args):
                    __args = _args[:len(args)]
                    __kwards = {name: k for name, k in zip(self.keys, _args[len(args):])}
                    return self.model.forward(*__args, **kwargs)
            mul, add = profile_mul_and_add(_M(self), args + tuple(kwargs.values()))
            batch = [i.shape[0] for i in args + tuple(kwargs.values()) if isinstance(i, torch.Tensor)][0]
            print("input shape: " + ' '.join(
                "{}".format(i.shape if isinstance(i, torch.Tensor) else i) for i in args) + ' '.join(
                "{}: {}".format(name, i.shape if isinstance(i, torch.Tensor) else i) for name, i in kwargs.items()))

            normal_parameters = 0
            ternary_parameters = 0
            for name, val in self.state_dict().items():
                if name.endswith('.u') or name.endswith('.v'):
                    ternary_parameters += val.numel()
                else:
                    normal_parameters += val.numel()
            print('parameters: {} M, tern p: {} M'.format(normal_parameters / 1e6, ternary_parameters / 1e6))
            print('mul: {} G, add: {} G'.format(mul / batch / 1e9, add / batch / 1e9))
            tested = True
        return _forward(self, *args, **kwargs)

    model.__class__.forward = forward
    return model
