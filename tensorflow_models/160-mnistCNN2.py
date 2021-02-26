"""
现代化的卷积神经网络：
* Batch Normalization
* leaky relu
* strided convolutioon

Tutorial on how to build a convnet w/ modern changes, e.g.
Batch Normalization, Leaky rectifiers, and strided convolution.

Parag K. Mital, Jan 2016.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def batch_norm(x, phase_train, scope='bn', affine=True):
    """
    Batch normalization on convolutional maps.

    from: https://stackoverflow.com/questions/33949786/how-could-i-
    use-batch-normalization-in-tensorflow

    Only modified to infer shape from input tensor x.

    Parameters
    ----------
    x
        Tensor, 4D BHWD input maps
    phase_train
        boolean tf.Variable, true indicates training phase
    scope
        string, variable scope
    affine
        whether to affine-transform outputs

    Return
    ------
    normed
        batch-normalized maps
    """
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()

        beta = tf.Variable(tf.constant(0.0, shape=[shape[-1]]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[shape[-1]]),
                            name='gamma', trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            """Summary

            Returns
            -------
            name : TYPE
                Description
            """
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema_mean, ema_var))
        epsilon = 1e-3  # 防止分母为0
        normed = tf.nn.batch_norm_with_global_normalization(
            x, mean, var, beta, gamma, epsilon, scale_after_normalization=affine)
    return normed


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


def conv2d(x, n_filters,
           k_h=5, k_w=5,
           stride_h=2, stride_w=2,
           stddev=0.02,
           activation=None,
           bias=True,
           padding='SAME',
           name="Conv2D"):
    """2D Convolution with options for kernel size, stride, and init deviation.

    Parameters
    ----------
    x : Tensor
        Input tensor to convolve.
    n_filters : int
        Number of filters to apply.
    k_h : int, optional
        Kernel height.
    k_w : int, optional
        Kernel width.
    stride_h : int, optional
        Stride in rows.
    stride_w : int, optional
        Stride in cols.
    stddev : float, optional
        Initialization's standard deviation.
    activation : arguments, optional
        Function which applies a nonlinearity
    padding : str, optional
        'SAME' or 'VALID'
    name : str, optional
        Variable scope to use.

    Returns
    -------
    x : Tensor
        Convolved input.
    """
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w', [k_h, k_w, x.get_shape()[-1], n_filters],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(
            x, w, strides=[1, stride_h, stride_w, 1], padding=padding)
        if bias:
            b = tf.get_variable('b', [n_filters], initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.bias_add(conv, b)
        if activation:
            conv = activation(conv)
        return conv


def linear(x, n_units, scope=None, stddev=0.02,
           activation=lambda x: x):
    """Fully-connected network.
    Parameters
    ----------
    x : Tensor
        Input tensor to the network.
    n_units : int
        Number of units to connect to.
    scope : str, optional
        Variable scope to use.
    stddev : float, optional
        Initialization's standard deviation.
    activation : arguments, optional
        Function which applies a nonlinearity
    Returns
    -------
    x : Tensor
        Fully-connected output.
    """
    shape = x.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], n_units], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        return activation(tf.matmul(x, matrix))


# %% Setup input to the network and true output label.  These are
# simply placeholders which we'll fill in later.
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# %% We add a new type of placeholder to denote when we are training.
# This will be used to change the way we compute the network during
# training/testing.
# 这种方式非常好，在构图上实现了复用
is_training = tf.placeholder(tf.bool, name='is_training')

# %% We'll convert our MNIST vector data to a 4-D tensor:
# N x W x H x C
x_tensor = tf.reshape(x, [-1, 28, 28, 1])

# %% We'll use a new method called  batch normalization.
# This process attempts to "reduce internal covariate shift"
# which is a fancy way of saying that it will normalize updates for each
# batch using a smoothed version of the batch mean and variance
# The original paper proposes using this before any nonlinearities
h_1 = lrelu(batch_norm(conv2d(x_tensor, 32, name='conv1'),
                       is_training, scope='bn1'), name='lrelu1')
h_2 = lrelu(batch_norm(conv2d(h_1, 64, name='conv2'),
                       is_training, scope='bn2'), name='lrelu2')
h_3 = lrelu(batch_norm(conv2d(h_2, 64, name='conv3'),
                       is_training, scope='bn3'), name='lrelu3')
h_3_flat = tf.reshape(h_3, [-1, 64 * 4 * 4])
h_4 = linear(h_3_flat, 10)
y_pred = tf.nn.softmax(h_4)

# %% Define loss/eval/training functions
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# %% We now create a new session to actually perform the initialization the
# variables:
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# %% We'll train in minibatches and report accuracy:
n_epochs = 10
batch_size = 100
for epoch_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={
            x: batch_xs,
            y: batch_ys,
            is_training: True
        })
    print(sess.run(accuracy, feed_dict={
        x: mnist.validation.images,
        y: mnist.validation.labels,
        is_training: False
    }))
