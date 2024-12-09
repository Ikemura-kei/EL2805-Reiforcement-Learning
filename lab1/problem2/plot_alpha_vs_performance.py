import numpy as np
import matplotlib.pyplot as plt

data = np.load("./perf_vs_alpha.npy")
alphas = data[:,0]
means = data[:,1]
stds = data[:,2]


# plt.errorbar(alphas, means, stds, ecolor='r', marker='o', markersize=3)
plt.fill_between(alphas, means-stds, means+stds, color='C0', alpha=0.5)
plt.plot(alphas, means, 'g')
plt.title('Performance v.s. alpha')
plt.xlabel('Alpha')
plt.ylabel('Average evaluation reward')
plt.grid(True)
plt.savefig('Perf_vs_alpha.png')
plt.show()