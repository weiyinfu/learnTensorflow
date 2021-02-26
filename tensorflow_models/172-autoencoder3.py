"""Tutorial on how to create an autoencoder w/ Tensorflow.

多层全连接编码-解码器

Parag K. Mital, Jan 2016
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


def autoencoder(dimensions=(784, 256, 64)):
    """Build a deep autoencoder w/ tied weights.

    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # %% input to the network
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    current_input = x

    # %% Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):  # 除去输入层
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(tf.random_uniform([n_input, n_output],
                                          -1.0 / math.sqrt(n_input),
                                          1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    # %% latent representation
    z = current_input
    encoder.reverse()  # 开始解码

    # %% Build the decoder using the same weights
    """
    编码器解码器只共享权值,不共享偏移
    """
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    # %% now have the reconstruction through the network
    y = current_input

    # %% cost function measures pixel-wise difference
    # 对于编码器解码器往往使用二范数作为误差,而不是使用交叉熵
    cost = tf.reduce_sum(tf.square(y - x))
    return {'x': x, 'z': z, 'y': y, 'cost': cost}


def test_mnist():
    """Test the autoencoder using MNIST."""

    # %%
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder()

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    batch_size = 50
    n_epochs = 10
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            # 训练的时候把数据中心化一下
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    # Plot example reconstructions
    n_examples = 15
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape([recon[example_i, :] + mean_img], (28, 28)))
        axs[0][example_i].axis('off')
        axs[1][example_i].axis('off')
    fig.show()
    plt.draw()
    plt.show()


if __name__ == '__main__':
    test_mnist()
