import torch
import torch.nn.functional as F
import numpy as np
from tern_svd.ternary_decompose_pytorch import ternary_decomposition, check
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

bit=32

def unfold(kernel, tile, stride, dilation):
    inputs = []
    for i in range(kernel.shape[0] * tile[0] * tile[1]):
        out = torch.zeros([kernel.shape[0] * tile[0] * tile[1]], device=kernel.device)
        out[i] = 1
        out = out.reshape([1, kernel.shape[0], tile[0], tile[1]])
        input = F.conv_transpose2d(input=out, weight=kernel, padding=(0, 0), output_padding=(0, 0), stride=stride, dilation=dilation)
        inputs.append(input.reshape([-1]))

    inputs = torch.stack(inputs, dim=0)
    return inputs

def test_kernel(kernel_size, tile, stride, dilation, tol):
    kernel = torch.distributions.Gamma(1, 1).sample([1, 1, kernel_size[0], kernel_size[1]]).cuda()
    kernel = torch.copysign(kernel, torch.randn(kernel.shape, device=kernel.device))
    kernel = unfold(kernel, tile, stride, dilation)[None, :, :]

    u, s, v = ternary_decomposition(kernel, tolerance=tol, cos_thresh=0.8386, stride=1)
    error, rank, max_rank, sparsity = check(kernel, u, s, v, bits=bit)
    return float(error.cpu()), float(rank / max_rank)


fig = plt.figure(figsize=(5, 5*0.75))
ax = fig.add_subplot(1, 1, 1)

for marker, tile in {'^': (6, 6),
                    'v': (3, 3),
                    'D': (2, 2),
                    'o': (1, 1),
                    }.items():
    for color, (kernel, stride, dilation) in {
                                     'r': [(3, 3), (1, 1), (1, 1)],
                                     'y': [(3, 3), (2, 2), (2, 2)],
                                     'b': [(4, 4), (2, 2), (1, 1)],
                                     'c': [(7, 7), (3, 3), (1, 1)]
                                               }.items():
        label =  "t{}k{}s{}d{}".format(tile[0], kernel[0], stride[0], dilation[0])
        print(label)
        notes = []
        for tol in [1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1]:
            error, compress = test_kernel(kernel, tile, stride, dilation, tol)
            notes.append((error, compress))
        notes = np.array(notes).T
        if tile == (1, 1):
            ax.plot(notes[1], notes[0], '--', marker=marker, color=color)
        else:
            ax.scatter(notes[1], notes[0], marker=marker, color=color)

custom_lines = [Patch(facecolor='r', label='k3s1d1'),
                Patch(facecolor='y', label='k3s2d2'),
                Patch(facecolor='b', label='k4s2d1'),
                Patch(facecolor='c', label='k7s3d1'),
                Line2D([0], [0], marker='o', color='k', linestyle='dashed', label='tile_1x1'),
                Line2D([0], [0], marker='D', color='k', linestyle='None', label='tile_2x2'),
                Line2D([0], [0], marker='v', color='k', linestyle='None', label='tile_3x3'),
                Line2D([0], [0], marker='^', color='k', linestyle='None', label='tile_6x6'),
                Line2D([0], [0], marker='D', color='r', linestyle='None', markeredgecolor='c', label='F(2x2, 3x3)')]

ax.set_yscale('log')
ax.set_xlabel('compress rate')
ax.set_ylabel('error')
ax.legend(handles=custom_lines)
f23 = (16*bit + 32 + 24)/(36*bit)
ylim = ax.get_ylim()
print(ylim)
ax.scatter([f23], [ylim[0]], marker='D', color='r',  edgecolor='c', clip_on=False)
ax.set_ylim(ylim)
fig.savefig('profile_2.png', dpi=300, bbox_inches='tight', pad_inches=0)

