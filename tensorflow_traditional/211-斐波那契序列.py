import tensorflow as tf

"""
求第n个斐波那契数字是多少
"""

n = tf.constant(15)


def cond(i, a, b):
    return i < n


def body(i, a, b):
    return i + 1, b, a + b


i, a, b = tf.while_loop(cond, body, (2, 1, 1))

print(tf.Session().run(b))
