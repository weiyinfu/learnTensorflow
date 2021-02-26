import tensorflow as tf
import numpy as np

"""
本节主要讲tf.where的用法
还是以FizzBuzz为例进行讲解
tf.where接受三个等长的向量，最终输出也是等长的向量
tf.where相当于条件表达式
tf.where(boolean_vector,if_true_vector,if_false_vector)
"""
n = 20
a = tf.Variable(np.arange(n))
wh = tf.where(
    tf.logical_and(tf.equal(tf.mod(a, 3), 0), tf.equal(tf.mod(a, 5), 0)),
    [tf.constant("FizzBuzz")] * n,
    tf.where(
        tf.logical_and(tf.not_equal(tf.mod(a, 3), 0), tf.equal(tf.mod(a, 5), 0)),
        [tf.constant('Buzz')] * n,
        tf.where(
            tf.logical_and(tf.equal(tf.mod(a, 3), 0), tf.not_equal(tf.mod(a, 5), 0)),
            ["Fizz"] * n,
            tf.as_string(a))))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ans = sess.run(wh)
    print(type(ans))
    print([str(i, 'utf8') for i in ans])
