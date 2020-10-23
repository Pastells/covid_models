# Comparing exponential and Erlang distributions
import numpy as np
import matplotlib.pyplot as plt

def poisson_time(x):
    # Time intervals of a Poisson process follow an exponential distribution
    return -np.log(1-np.random.random())/x

size = int(1e6)
n_bins=int(1e3)
x = np.zeros(size)
y = np.zeros(size)
for i in range(size):
    x[i] = poisson_time(1)
    y[i] = poisson_time(1)+poisson_time(1)

plt.hist(x,bins=n_bins)
plt.hist(y,bins=n_bins)
plt.show()
