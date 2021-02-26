import numpy as np
import tensorflow as tf
"""
embedding_lookup其实就是查表
"""
a = np.tile(np.arange(0, 5).reshape(-1, 1), (1, 5))
idx1 = tf.Variable([0, 2, 3, 1], tf.int32)
idx2 = tf.Variable([[0, 2, 3, 1], [4, 0, 2, 2]], tf.int32)
out1 = tf.nn.embedding_lookup(a, idx1)
out2 = tf.nn.embedding_lookup(a, idx2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(out1))
    print(out1)
    print(sess.run(out2))
    print(out2)
