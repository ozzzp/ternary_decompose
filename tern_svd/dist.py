from queue import PriorityQueue

import torch
from torch import distributed as dist

__all__ = ['sync_tern_svd_layer', 'dispatch']

def sync_tern_svd_layer(M, src):
    _sync(M, 'u', src)
    _sync(M, 's', src)
    _sync(M, 'v', src)
    _sync(M, 'weight', src)
    _sync(M, 'rest_weight', src)


def _sync(M, name, src):
    if hasattr(M, name):
        device = getattr(M, name).device
        dtype = getattr(M, name).dtype
        shape = torch.tensor(getattr(M, name).shape, dtype=torch.int64, device=device)
    else:
        device = M.weight.device
        if name in ['u', 'v']:
            dtype = torch.float32
            dim_shape = len(M.weight.shape)
        elif name in ['rest_weight']:
            dtype = M.weight.dtype
            dim_shape = len(M.weight.shape)
        elif name in ['s']:
            dtype = M.weight.dtype
            dim_shape = 1
        else:
            raise ValueError
        shape = torch.empty([dim_shape], dtype=torch.int64, device=device)
    dist.broadcast(shape, src)
    shape = tuple(shape.cpu().tolist())
    if not hasattr(M, name) or shape != getattr(M, name).shape:
        new_tensor = torch.empty(shape, device=device, dtype=dtype)
        if hasattr(M, name):
            delattr(M, name)
        M.register_buffer(name, new_tensor)
    dist.broadcast(getattr(M, name), src)


class dispatch:
    def __init__(self, word_size):
        self.queue = PriorityQueue()
        for i in range(word_size):
            self.queue.put((0, i))

    def get_rank(self, num):
        n, r = self.queue.get()
        self.queue.put((n+num, r))
        return r
