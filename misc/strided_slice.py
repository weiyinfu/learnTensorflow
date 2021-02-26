import tensorflow as tf
import numpy as np

a = tf.constant([1, 2, 3, 4])
one = tf.strided_slice(a, [0], [5], [2])
two = tf.slice(a, [1], [2])

a = tf.constant(np.arange(0, 24).reshape(4, 6))
with tf.Session()as sess:
    print(sess.run([one, two]))
    print(sess.run(a))
    print(sess.run([tf.strided_slice(a, [0, 0], [5, 5], [2, 2])]))
