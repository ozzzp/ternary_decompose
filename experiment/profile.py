import torch
import numpy as np
from tern_svd.ternary_decompose_pytorch import ternary_decomposition, check
import matplotlib.pylab as plt

A = torch.distributions.Gamma(1, 1).sample([256, 512]).cuda()
A = torch.copysign(A, torch.randn(A.shape, device=A.device))
fig = plt.figure(figsize=(5, 0.75*5))
ax = fig.add_subplot(1, 1, 1)

tern_svd_thresh = [1e-4, 1e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1]

def tern_svd_run(tol):
    u, s, v = ternary_decomposition(A[None, :, :], tolerance=tol, cos_thresh=0.8386, stride=5)
    error, rank, max_rank, sparsity = check(A, u, s, v, bits=32)
    return float(error.cpu()), float(rank / max_rank)

tern_svd_data = np.array([tern_svd_run(t) for t in tern_svd_thresh]).T
ax.plot(tern_svd_data[1], tern_svd_data[0], 'bo-', label='ternary svd')


svd_rank_thresh = np.arange(0, 1, 0.05)

def svd_run(rank_thresh):
    u, s, v = torch.linalg.svd(A)
    criticak_rank = (A.shape[0] * A.shape[1]) / (A.shape[0] + A.shape[1])
    rank = int(rank_thresh * criticak_rank)
    return float((s[rank] / s[0]).cpu()), rank / criticak_rank

svd_data = np.array([svd_run(t) for t in svd_rank_thresh]).T
ax.plot(svd_data[1], svd_data[0], 'y+--', label='svd')

sparsity_thresh = np.arange(0.05, 1.05 , 0.05)

def get_norm(A):
    _, s, _ = torch.linalg.svd(A)
    return s[0]

def sparsity_run(thresh):
    A_data, _ = torch.sort(torch.abs(A).reshape([-1]))
    thresh = A_data[int(A_data.numel() * thresh)] if thresh < 1 else float('Inf')
    A_sparse = torch.where(torch.abs(A) >= thresh, A, torch.zeros_like(A))
    return float((get_norm(A - A_sparse)/get_norm(A)).cpu()), float(torch.count_nonzero(A_sparse) / A.numel())

sparsity_data = np.array([sparsity_run(t) for t in sparsity_thresh]).T
ax.plot(sparsity_data[1], sparsity_data[0], 'rx--', label='sparsity')

quant_bit = [2, 4, 8, 12, 16]

def quant_run(bit):
    mean = torch.mean(A)
    scale = torch.abs(A - mean).max()
    qA = (A - mean) / scale
    bit_scale = (2 ** (bit - 1) - 1)
    qA = torch.round(qA * bit_scale)
    qA /= bit_scale
    qA = scale * qA + mean

    compress_rate = (1 + (bit - 2)) / (1 + 32 - 2)
    return float((get_norm(A - qA) / get_norm(A)).cpu()), compress_rate


quant_data = np.array([quant_run(t) for t in quant_bit]).T
ax.plot(quant_data[1], quant_data[0], 'g2--', label='quantization')

for bit, y, x in zip(quant_bit, quant_data[0], quant_data[1]):
    plt.annotate("qint{}".format(bit), (x, y))

ax.set_yscale('log')
ax.set_xlabel('compress rate')
ax.set_ylabel('error')
ax.legend()
fig.savefig('profile.png', dpi=300, bbox_inches='tight', pad_inches=0)

