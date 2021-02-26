import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

"""
画高斯分布
"""
sigma = 1.0
mean = 0.0
n_values = 32
sess = tf.InteractiveSession()
x = tf.linspace(-3.0, 3.0, n_values)
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                        (2.0 * tf.pow(sigma, 2.0)))) *
     (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
plt.plot(z.eval())
plt.show()

print(tf.shape(z).eval(), z.shape, z.get_shape(), z.get_shape().as_list())

z_2d = tf.matmul(tf.reshape(z, [np.prod(z.shape), 1]), tf.reshape(z, [1, np.prod(z.shape)]))
plt.imshow(z_2d.eval())
plt.show()
# %% For fun let's create a gabor patch:
x = tf.reshape(tf.sin(x), [n_values, 1])
y = tf.reshape(tf.ones_like(x), [1, n_values])
z = tf.multiply(tf.matmul(x, y), z_2d)
plt.imshow(z.eval())
plt.show()

# We can also list all the operations of a graph:
ops = tf.get_default_graph().get_operations()
print([op.name for op in ops])


# Lets try creating a generic function for computing the same thing:
def gabor(n_values=32, sigma=1.0, mean=0.0):
    x = tf.linspace(-3.0, 3.0, n_values)
    z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                            (2.0 * tf.pow(sigma, 2.0)))) *
         (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
    gauss_kernel = tf.matmul(
        tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))
    x = tf.reshape(tf.sin(tf.linspace(-3.0, 3.0, n_values)), [n_values, 1])
    y = tf.reshape(tf.ones_like(x), [1, n_values])
    gabor_kernel = tf.multiply(tf.matmul(x, y), gauss_kernel)
    return gabor_kernel


# %% Confirm this does something:
plt.imshow(gabor().eval())
plt.show()


# %% And another function which can convolve
def convolve(img, W):
    # The W matrix is only 2D
    # But conv2d will need a tensor which is 4d:
    # height x width x n_input x n_output
    if len(W.get_shape()) == 2:
        dims = W.get_shape().as_list() + [1, 1]
        W = tf.reshape(W, dims)

    if len(img.get_shape()) == 2:  # 如果是灰度图
        # num x height x width x channels
        dims = [1] + img.get_shape().as_list() + [1]
        img = tf.reshape(img, dims)
    elif len(img.get_shape()) == 3:
        # 如果是彩色图num*width*height*channel
        dims = [1] + img.get_shape().as_list()
        img = tf.reshape(img, dims)
        # if the image is 3 channels, then our convolution
        # kernel needs to be repeated for each input channel
        W = tf.concat(axis=2, values=[W, W, W])

    # Stride is how many values to skip for the dimensions of
    # num, height, width, channels
    convolved = tf.nn.conv2d(img, W, strides=[1, 1, 1, 1], padding='SAME')
    return convolved


# %% Load up an image:
from skimage import data

img = data.astronaut()
plt.imshow(img)
plt.show()
print(img.shape)

# %% Now create a placeholder for our graph which can store any input:
x = tf.placeholder(tf.float32, shape=img.shape)

out = convolve(x, gabor())

# %% Now send the image into the graph and compute the result
result = tf.squeeze(out).eval(feed_dict={x: img})
plt.imshow(result)
plt.show()
