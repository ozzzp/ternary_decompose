import torch
from contextlib import contextmanager
from unittest.mock import patch
from .tern_svd_linear import Ternary_SVD_Linear
from .tern_svd_conv2d import Ternary_SVD_Conv2D
from .tern_svd_conv2d_transpose import Ternary_SVD_ConvTranspose
from .ternary_decompose_pytorch import ternary_decomposition, check, _get_critical_rank, get_s

__all__ = ['replace_Linear_to_ternary_SVD_linear', 'Ternary_SVD_Layer', 'transform_policy', 'transform_policy_for_QAT', 'tern_svd_layer_patch']

@contextmanager
def replace_Linear_to_ternary_SVD_linear():
    with patch.object(torch.nn, "Linear", Ternary_SVD_Linear):
        with patch.object(torch.nn, "Conv2d", Ternary_SVD_Conv2D):
            with patch.object(torch.nn, "ConvTranspose2d", Ternary_SVD_ConvTranspose):
                yield

Ternary_SVD_Layer = (Ternary_SVD_Linear, Ternary_SVD_Conv2D, Ternary_SVD_ConvTranspose)

def transform_policy(steps=20, tolerance=5e-2, cos_thresh=0.8386, bits=8, verbose=False):
    def transform(X):
        M, N = X.shape[-2:]
        stride = (M * N) / (M + N) / steps
        stride = max(1, int(stride))
        U, S, V = ternary_decomposition(X, tolerance=tolerance, cos_thresh=cos_thresh, stride=stride)
        error, rank, max_rank, sparsity = check(X, U, S, V, bits=8)
        _, _, max_rank_2, _ = check(X, U, S, V, never_mind_sparsity=True, bits=bits)
        srange = torch.max(torch.abs(S).max(dim=-1)[0]/torch.abs(S).min(dim=-1)[0])
        if verbose:
            print("error: {:.3g}, rank: {}/{}/{}, cost:{:.3g}/{:.3g}, sparsity: {:.3g}, srange: {:.3g}, smin: {:.3g}".format(
                error,
                int(rank),
                int(max_rank),
                int(max_rank_2),
                rank / max_rank,
                rank / max_rank_2,
                sparsity,
                srange,
                torch.abs(S).min()
            ))

        return rank / max_rank, U, S, V
    return transform

def transform_policy_for_QAT(steps=20, tolerance=5e-2, cos_thresh=0.8386, bits=8, verbose=False):
    def _transform(X, pre_u=None, pre_v=None):
        M, N = X.shape[-2:]
        stride = (M * N) / (M + N) / steps
        stride = max(1, int(stride))

        U, S, V = ternary_decomposition(X, pre_u=pre_u, pre_v=pre_v, tolerance=tolerance, cos_thresh=cos_thresh, stride=stride)
        error, rank, max_rank, sparsity = check(X, U, S, V, bits=8)
        if verbose:
            _, _, max_rank_2, _ = check(X, U, S, V, never_mind_sparsity=True, bits=bits)
            srange = torch.max(torch.abs(S).max(dim=-1)[0] / torch.abs(S).min(dim=-1)[0])
            print("error: {:.3g}, rank: {}/{}/{}, cost:{:.3g}/{:.3g}, sparsity: {:.3g}, srange: {:.3g}".format(
                error,
                int(rank),
                int(max_rank),
                int(max_rank_2),
                rank / max_rank,
                rank / max_rank_2,
                sparsity,
                srange))

        return error, U, S, V

    def _get_s(A, U, V):
        return get_s(A, U, V, cos_thresh=cos_thresh)

    return _transform, _get_s

def tern_svd_layer_patch(fun):
    def _fun(M):
        if isinstance(M, Ternary_SVD_Layer):
            fun(M)

    return _fun
