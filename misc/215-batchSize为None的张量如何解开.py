"""
输入一个张量x_place，它的形状是None，1
使用map_fn可以无需关心形状直接映射
"""
import tensorflow as tf
import numpy as np

x_place = tf.placeholder(tf.int32, (None, 1))


def add_each(x_place):
    # 为列表中的每个数字增加1
    def add_one(x):
        return x + 1

    y = tf.map_fn(lambda x: add_one(x), x_place)
    return y


y = add_each(x_place)
with tf.Session() as sess:
    x = np.random.randint(0, 10, (np.random.randint(2, 5))).reshape(-1, 1)
    print(x.reshape(-1))
    print(sess.run(y, feed_dict={
        x_place: x
    }).reshape(-1))
