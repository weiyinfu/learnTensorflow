import tensorflow as tf
import numpy as np
import pylab as plt

"""
softmax能够起到平滑的作用
softsign也能
"""
x = np.linspace(-100, 100, 100)
y = tf.nn.softsign(tf.constant(x))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.plot(x, sess.run(y), label='tf-softsign')
    plt.plot(x, x / (1 + np.abs(x)), label='np-softsign')
    plt.legend()
    plt.show()
