import torch

class TernSVDBase(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super(TernSVDBase, self).__init__()

        def regiter_svd_buffer(self, state_dict, prefix):
            for name in ['s', 'u', 'v', 'rest_weight']:
                key = prefix + name
                if key in state_dict:
                    self.register_buffer(name, torch.empty_like(state_dict[key]).to(self.weight.device))