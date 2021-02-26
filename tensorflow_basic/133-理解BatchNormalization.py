import tensorflow as tf
import numpy as np

a = tf.constant(np.random.randint(0, 10, (4, 2, 2, 3)), dtype=tf.float32)
"""
moments意思是求矩：一阶中心矩是均值
二阶中心矩是方差

tf.nn.batch_norm_with_global_normalization已经不鼓励使用了
"""
mu, sigma = tf.nn.moments(a, axes=[0, 1, 2])
offset = tf.Variable(2, dtype=tf.float32)
scale = tf.Variable(3, dtype=tf.float32)
epsilon = 1e-3
normed_a = tf.nn.batch_normalization(a, mu, sigma, offset, scale, epsilon)


def mynorm(a, mean, var, shift, scale, epsilon):
    aa = (a - mean) / (var + epsilon) ** 0.5
    return scale * aa + shift


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a_value, mean, var, normed_a_value = sess.run([a, mu, sigma, normed_a])
    print(a_value)
    print('=' * 10)
    print(mean)
    print('=' * 10)
    print(var)
    print('=' * 10)
    print(normed_a_value)
    print('=' * 10)
    print(mynorm(a_value, mean, var, offset.eval(), scale.eval(), epsilon))
