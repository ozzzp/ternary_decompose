import warnings

import torch

from .handlers import handlers
from .utils.trace import trace
import functools

__all__ = ['profile_mul_and_add']

def trace_back_tensor(graph, t):
    for node in graph.nodes:
        if t in node.outputs:
            assert len(node.outputs) == 1
            return trace_back_tensor(graph, node.inputs[0])
    return t

def profile_mul_and_add(model, args=(), kwargs=None, reduction=sum):
    results = dict()
    input_length = len(args) if isinstance(args, (tuple, list)) else 1

    graph = trace(model, args, kwargs)
    parameters = list(model.state_dict().values())
    for node in graph.nodes:
        for operators, func in handlers:
            if isinstance(operators, str):
                operators = [operators]
            if node.operator in operators:
                if func is not None:
                    results[node] = func(node)
                break
        else:
            warnings.warn('No handlers found: "{}". Skipped.'.format(
                node.operator))

    if reduction is not None:
        def reduce(results):
            for out in results.values():
                if len(out) == 3:
                    mul, add, kernel = out
                    kernel = trace_back_tensor(graph, kernel)
                    p = parameters[int(kernel.name) - input_length]
                    assert tuple(kernel.shape) == tuple(p.shape), "{} vs {}".format(tuple(kernel.shape), tuple(p.shape))
                    mul_rate = 1 - torch.isin(p, torch.tensor([0, 1, -1], device=p.device)).count_nonzero() / p.numel()
                    add_rate = 1 - torch.isin(p, torch.tensor([0], device=p.device)).count_nonzero() / p.numel()
                    yield  int(mul * mul_rate), int(add * add_rate)
                elif len(out) == 2:
                    yield out

        return functools.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), reduce(results))
    else:
        return results
