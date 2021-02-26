# add_n的用法
import tensorflow as tf

"""
reduce_sum比add_n要灵活
"""
input1 = tf.constant([1.0, 2.0, 3.0])
input2 = tf.Variable(tf.random_uniform([3]))
input3 = tf.Variable(tf.random_uniform([3]))
output = tf.add_n([input1, input2, input3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(input1 + input2 + input3))
    print(sess.run(tf.reduce_sum(tf.concat([[input1], [input2], [input3]], axis=0), axis=0)))
    print(sess.run(tf.add(tf.add(input1, input2), input3)))
    print(sess.run(output))
