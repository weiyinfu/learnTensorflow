import tensorflow as tf

"""
使用tensorarray实现斐波那契数列
"""
n = tf.constant(5)

c = tf.TensorArray(tf.int32, n)
print(type(c))
c = c.write(0, 1)
print(type(c))
c = c.write(1, 1)


def cond(i, a, b, c):
    return i < n


def body(i, a, b, c):
    c = c.write(i, a + b)
    return i + 1, b, a + b, c


i, a, b, c = tf.while_loop(cond, body, (2, 1, 1, c))

c = c.stack()

print(tf.Session().run(c))
