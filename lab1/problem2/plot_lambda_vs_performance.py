import numpy as np
import matplotlib.pyplot as plt

data = np.load("./perf_vs_lambda.npy")
lmbdas = data[:,0]
means = data[:,1]
stds = data[:,2]


plt.fill_between(lmbdas, means-stds, means+stds, color='C0', alpha=0.5)
plt.plot(lmbdas, means, 'g')
plt.title('Performance v.s. lambda')
plt.xlabel('Lambda')
plt.ylabel('Average evaluation reward')
plt.grid(True)
plt.savefig('Perf_vs_lambda.png')
plt.show()