import tensorflow as tf

"""
什么是local_variable?我咋没见过
"""
a = tf.Variable(3, dtype=tf.float32, name='a')

b = tf.get_local_variable('b', shape=tuple(), dtype=tf.int32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(tf.global_variables())
    print(tf.local_variables())
    print(sess.run(a))
    print(sess.run(b))
