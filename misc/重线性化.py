"""
使用sigmoid的时候，激活函数采用y*log(sigmoid(x)),其中y为上一时刻的输出

"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 100)
y = np.log(1 / (1 + np.exp(-x)))
plt.plot(x, y)
plt.show()
