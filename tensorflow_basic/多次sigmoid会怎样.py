import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


import pylab as plt

x = np.linspace(-50, 50, 500)
plt.plot(sigmoid(x), label='$\sigma^1$')
plt.plot(sigmoid(sigmoid(x)), label='$\sigma^2$')
plt.plot(sigmoid(sigmoid(sigmoid(x))), label='$\sigma^3$')
plt.legend()
plt.title('what if simoid many times')
plt.show()
