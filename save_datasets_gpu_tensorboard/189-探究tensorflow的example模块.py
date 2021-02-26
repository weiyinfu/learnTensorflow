from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

"""
许多库都自带数据集，如skimage，sklearn等都自带了许多数据集
对常用数据集有所了解是很有必要的
"""
data = input_data.read_data_sets("MNIST_data", one_hot=1)
train = data.train.images
print(train)
fig, ax = plt.subplots(3, 3)
ax = ax.reshape(-1)
for i in range(9):
    ax[i].imshow(train[i].reshape(28, 28), cmap='gray')
    ax[i].axis('off')
plt.show()
