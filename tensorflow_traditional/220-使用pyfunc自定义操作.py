import numpy as np
import tensorflow as tf

"""
使用自定义函数，虽然运行效率会慢些，但是却获得了整个Python的灵活性

但是pyfunc却无法进行反向传播！网络中一旦假如pyfunc，梯度便传不过去了，就像绝缘体一样
"""
x = tf.placeholder(tf.float32, shape=[None])


def f(x):
    print(f.__name__, x, type(x))
    return np.sin(x)


y_mine = tf.py_func(f, [x], tf.float32)
y_true = tf.sin(x)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([y_mine, y_true], feed_dict={
        x: np.random.random(np.random.randint(3, 7))
    }))
