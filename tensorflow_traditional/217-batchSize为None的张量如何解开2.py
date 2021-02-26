"""
输入一个张量x_place，它的形状是None，1

使用while构造新的TensorArray
"""
import numpy as np
import tensorflow as tf

x_place = tf.placeholder(tf.int32, (None, 1))


def add_each(x_place):
    # 为列表中的每个数字增加1
    ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    def cond(i, ta, x):
        return i < tf.shape(x)[0]

    def body(i, ta, x):
        ta = ta.write(i, x[i] + 1)
        return i + 1, ta, x

    _, ta, _ = tf.while_loop(cond, body, [0, ta, x_place])
    return ta.stack()


y = add_each(x_place)
with tf.Session() as sess:
    x = np.random.randint(0, 10, (np.random.randint(2, 5))).reshape(-1, 1)
    print(x.reshape(-1))
    print(sess.run(y, feed_dict={
        x_place: x
    }).reshape(-1))
