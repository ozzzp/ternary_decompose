import torch
import torch.nn.functional as F
import numpy as np
from tern_svd.ternary_decompose_pytorch import ternary_decomposition, check
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

fig = plt.figure(figsize=(5, 5*0.75))
ax = fig.add_subplot(1, 1, 1)

data = np.random.randn(2, 400)
data = data[:, data[0] ** 2 + data[1] ** 2 <= 4] / 2 * 1.1
data[1] /= 3
theta = 60 / 180 * np.pi
transform_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])
data = transform_matrix @ data

ax.scatter(data[0], data[1])

top_dir = transform_matrix @ np.array([1, 0]).T * 1.1

print(top_dir)

ax.quiver([0], [0], top_dir[0], top_dir[1], color='r', angles='xy', scale_units='xy', scale=1, label='top component')
ax.quiver([0], [0], [1], [1], color='y', angles='xy', scale_units='xy', scale=1, label='top ternary component')

ax.set_aspect('equal')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.legend(loc=4)
fig.savefig('tsvd.png', dpi=300, bbox_inches='tight', pad_inches=0)