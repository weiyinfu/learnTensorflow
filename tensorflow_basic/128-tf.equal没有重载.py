import tensorflow as tf

a = tf.constant(3)
b = tf.constant(3)
c = tf.cast(a == b, tf.float32)
d = tf.cast(tf.equal(a, b), tf.float32)
with tf.Session() as sess:
    print(sess.run([c, d]))
