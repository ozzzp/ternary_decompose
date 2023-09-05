import matplotlib.pylab as plt
import numpy as np
import pickle

fig = plt.figure(figsize=(5, 0.75*5))
path = "runs/error_analysis_log.txt"
M = 256
N = 512
d = 32

sparsity_data = []

with open(path, 'r') as f:
    for line in f:
        l = line.split()
        s = float(l[1].rstrip(','))
        c = float(l[3].rstrip(','))/3.25
        e = float(l[5])
        if s >=24 and s<=60:
            sparsity_data.append((s, c, e))

sparsity_data = np.array(sparsity_data)


tricks = np.log(np.array([2.5e-5, 1e-4, 1e-3, 1e-2, 5e-2, 2e-1, 6e-1]))
label = ["0.0025%", "0.01%", "0.1%", "1%", "5%", "20%", "60%"]
plt.tricontourf(sparsity_data[:, 1], sparsity_data[:, 0], np.log(sparsity_data[:, 2]), cmap='jet_r', levels=1000)
plt.xlabel("compression rate")
plt.ylabel("$\\theta$(degree)")
cbar=plt.colorbar(ticks=tricks)
cbar.ax.set_yticklabels(label) # add the labels

cr = plt.tricontour(sparsity_data[:, 1], sparsity_data[:, 0], np.log(sparsity_data[:, 2]), tricks, colors='k')
plt.clabel(cr, inline=True, fmt={n:s for n, s in zip(cr.levels, label)}, fontsize=8)


plt.plot(np.arange(*plt.xlim(), 0.01), 33 * np.ones_like(np.arange(*plt.xlim(), 0.01)), '-.', label=('$\\theta = 33^{\\circ}, r=0.29 $'), color='k')
plt.legend(loc=3)
plt.savefig('error_2.png', dpi = 500, bbox_inches='tight', pad_inches=0)
