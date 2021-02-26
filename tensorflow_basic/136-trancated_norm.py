import tensorflow as tf
import matplotlib.pyplot as plt

"""
truncated_normal的随机数取值范围是 [u-2sigma,u+2sigma]
"""
a = tf.Variable(tf.truncated_normal((1000,), 0, stddev=2.0))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a_value = sess.run(a)
    plt.hist(a_value, bins=100)
    plt.show()
