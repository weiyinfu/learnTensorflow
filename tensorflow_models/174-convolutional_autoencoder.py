"""Tutorial on how to create a convolutional autoencoder w/ Tensorflow.

自动编码器是一种思想
它前面可以接CNN或者RNN

Parag K. Mital, Jan 2016
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


def corrupt(x):
    """Take an input tensor and add uniform masking.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.

    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    """
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                                    minval=0,
                                                    maxval=2,
                                                    dtype=tf.int32), tf.float32))


def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.

    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.

    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


# %%
def autoencoder(input_shape=(None, 784),
                n_filters=(1, 10, 10, 10),  # 深度
                filter_sizes=(3, 3, 3, 3),  # 过滤器的尺寸
                corruption=False):  # corruption表示是否对原图片加噪声
    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description

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

    Raises
    ------
    ValueError
        Description
    """
    x = tf.placeholder(tf.float32, input_shape, name='x')

    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        # 如果每个样本一个一维向量,reshape一下
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        # 如果每个样本是一个RGBA图像
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # Optionally apply denoising autoencoder
    if corruption:
        current_input = corrupt(current_input)

    # Build the encoder
    encoder = []
    shapes = []  # 反卷积过程的shape是一到多映射,所以必须记录下shape来
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]  # 当前深度
        shapes.append(current_input.get_shape().as_list())  # 记录当前shape
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x_tensor))

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost}


# %%
def test_mnist():
    """Test the convolutional autoencder using MNIST."""
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder()

    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    batch_size = 100
    n_epochs = 10
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    # Plot example reconstructions
    n_examples = 10
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (784,)) + mean_img,
                (28, 28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()


if __name__ == '__main__':
    test_mnist()
