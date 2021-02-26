"""

利用tensorflow的并行，能够快速计算parzen窗

"""
import matplotlib.pyplot as plt
import tensorflow as tf

tf.set_random_seed(0)
a = tf.Variable(tf.random_normal((5,), dtype=tf.float32))
x = tf.linspace(-2.0, 2.0, 1000)


def exponent(x):
    """
    指数分布parzen窗
    :param dis:
    :return:
    """
    return tf.exp(-x)


def gauss(x):
    """
    高斯分布parzen窗
    :param dis:
    :return:
    """
    return tf.exp(-x * x)


use = gauss

y = tf.map_fn(lambda xi: tf.reduce_sum(use(tf.abs(xi - a))), x)
ys = tf.map_fn(lambda ai: use(tf.abs(x - ai)), a)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a_value, x_value, y_value, ys_value = sess.run([a, x, y, ys])
    plt.plot(x_value, y_value)
    for y_now in ys_value:
        plt.plot(x_value, y_now)
    plt.show()
