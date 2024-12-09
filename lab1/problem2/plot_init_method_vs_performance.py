import numpy as np
import matplotlib.pyplot as plt

data = np.load("./perf_vs_init_q_std.npy")
std_used = data[:,0]
means = data[:,1]
stds = data[:,2]


# plt.errorbar(alphas, means, stds, ecolor='r', marker='o', markersize=3)
plt.fill_between(std_used, np.array([-114.92]* len(means)) - 12.547254679809445, np.array([-114.92]* len(means)) + 12.547254679809445, color='C3', alpha=0.3)
plt.fill_between(std_used, means-stds, means+stds, color='C0', alpha=0.5)
plt.plot(std_used, np.array([-114.92]* len(means)), 'r')
plt.plot(std_used, means, 'g')
plt.legend(['All zeros init', 'Gaussian with different stds'])
plt.title('Performance v.s. intialization method')
plt.xlabel('Std used, for green line only')
plt.ylabel('Average evaluation reward')
plt.grid(True)
plt.savefig('Perf_vs_init_method.png')
plt.show()