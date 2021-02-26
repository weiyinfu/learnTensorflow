import tensorflow as tf
import tensorflow.bitwise as bit

"""
是时候读一下tensorflow源码了
tensorflow.bitwise包下存放了许多位运算操作
"""
x = tf.placeholder(dtype=tf.int32, shape=(None,))
# 位运算只适用于整数
# y = tf.placeholder(dtype=tf.float32, shape=(None,))
y = tf.placeholder(dtype=tf.int32, shape=(None,))
and_op = bit.bitwise_and(x, y)
or_op = bit.bitwise_or(x, y)
xor_op = bit.bitwise_xor(x, y)
inverse_op = bit.invert(x)
left_shift = bit.left_shift(x, 3)
with tf.Session() as sess:
    print(sess.run([and_op, or_op, xor_op, inverse_op, left_shift], feed_dict={
        x: [1, 2, 3],
        y: [4, 5, 6]
    }))
