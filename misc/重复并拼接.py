import numpy as np
import tensorflow as tf

x = tf.placeholder(dtype=tf.int32, shape=(None, None, 3))
y = tf.placeholder(dtype=tf.int32, shape=(None, 3))

"""
how to repeat y and concat with x
"""
# 第一种方式
reshaped = tf.reshape(x, (-1, x.get_shape().as_list()[-1]))
z = tf.concat([reshaped, tf.tile(y, (tf.shape(reshaped)[0], 1))], axis=1)
z = tf.reshape(z, (tf.shape(x)[0], -1, x.get_shape().as_list()[-1] * 2))
# 第二种方式
repeat_y = tf.tile(tf.expand_dims(y, axis=1, ), (1, tf.shape(x)[1], 1))
zz = tf.concat([x, repeat_y], axis=2)
print(z.shape)
print(zz.shape)
with tf.Session()as sess:
    z_value, zz_value, x_value, y_value = sess.run([z, zz, x, y], feed_dict={
        x: np.random.randint(0, 9, (1, 2, 3)),
        y: np.random.randint(0, 9, (1, 3)),
    })
    print(x_value)
    print(y_value)
    print('=' * 10)
    print(z_value)
    print(zz_value)
