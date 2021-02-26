"""
已知概率密度函数生成分布,是一个非常有趣的问题
使用MullerBox算法生成高斯分布
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

N = 10000
u1 = tf.random_uniform((N,), dtype=tf.float32)
u2 = tf.random_uniform((N,), dtype=tf.float32)
R = tf.sqrt(-2 * tf.log(u1))
theta = 2 * np.pi * u2
# 两种拼接方式
# a = tf.reshape(tf.stack([R * tf.sin(theta), R * tf.cos(theta)]), (-1,))
a = tf.concat(values=[R * tf.sin(theta), R * tf.cos(theta)], axis=0)
with tf.Session() as sess:
    aa = sess.run(a)
    print(aa)
    plt.hist(aa, bins=1000)
    plt.show()
