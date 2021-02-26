"""
给定一系列总和和方差，求均值和标准差

ss表示充分统计量：sufficient statistics
"""
import numpy as np
import tensorflow as tf

group_count = 10
couts = np.random.randint(3, 5, group_count)
mean_ss = np.random.random(group_count)
var_ss = np.random.random(group_count)
mean, variance = tf.nn.normalize_moments(counts=tf.constant(couts, dtype=tf.float32), mean_ss=tf.constant(mean_ss, dtype=tf.float32), variance_ss=tf.constant(var_ss, dtype=tf.float32), shift=None)


def normalize_moments(counts, mean_ss, variance_ss, shift):
    if shift is None:
        shift = tf.constant(0.0, dtype=mean_ss.dtype)
    mean = mean_ss / counts + shift
    variance = variance_ss / counts - mean * mean
    return mean, variance


mymean, myvariance = normalize_moments(counts=tf.constant(couts, dtype=tf.float32), mean_ss=tf.constant(mean_ss, dtype=tf.float32), variance_ss=tf.constant(var_ss, dtype=tf.float32), shift=None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    m, v = sess.run([mean, variance])
    print(m, v)
    print(sess.run([mymean, myvariance]))
