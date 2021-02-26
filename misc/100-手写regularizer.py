import tensorflow as tf
import numpy as np

REGULARIZATION_RATE = 1.0
regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
x = tf.Variable([1, 2, 3, 4], dtype=tf.float32)
y = tf.Variable([4, 5, 6, 7], dtype=tf.float32)
tf.add_to_collection('weight', x)
tf.add_to_collection('weight', y)
# collection是语法糖
for i in tf.get_collection('weight'):
    z = regularizer(i)
    tf.add_to_collection('regularize', z)
loss = tf.add_n(tf.get_collection('regularize'))


def myregular(collection):
    for i in tf.get_collection("weight"):
        tf.add_to_collection("regu", tf.reduce_sum(i ** 2) * 0.5)
    return tf.add_n(tf.get_collection('regu'))


loss_mine = myregular('weight')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    xx, yy, ll = sess.run([x, y, loss])
    print(xx, yy, ll)
    print((np.sum(xx ** 2) + np.sum(yy ** 2)) * 0.5)
    print(sess.run(tf.get_collection('regularize')))
    print(sess.run(loss_mine))
