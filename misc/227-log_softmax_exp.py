import numpy as np


def softmax(x):
    return 1 / (1 + np.exp(-x))


x = np.linspace(-10, 20, 100)
import matplotlib.pyplot as plt

plt.plot(x, np.log(softmax(np.exp(x))), label='logsoftmax',linewidth=2)
plt.plot(x, softmax(x), label='softmax',linewidth=2)
plt.legend()
plt.show()
