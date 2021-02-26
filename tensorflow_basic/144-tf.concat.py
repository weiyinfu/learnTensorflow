import tensorflow as tf

a = tf.constant([[1, 2, ], [3, 4]], dtype=tf.float32)
horizon = tf.concat([a, a], axis=0)
vertical = tf.concat([a, a], axis=1)

b = tf.constant(2, dtype=tf.float32)

"""
tf.tile(a,shape)
a的rank必须和shape的rank一致
"""
concat_one = tf.concat([a, tf.tile(tf.reshape(b, (-1, 1)), (a.shape[0], 1))], axis=1)
with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(horizon))
    print(sess.run(vertical))
    print(sess.run(tf.tile(a, (2, 2))))
    print(sess.run(concat_one))
