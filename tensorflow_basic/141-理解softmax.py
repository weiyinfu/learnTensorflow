"""
softmax的本质是：使向量满足
* 非负性
* 归一性
"""
import tensorflow as tf
import numpy as np

x = np.array([1, 2, 3])
a = tf.Variable(x, dtype=tf.float32)
b = tf.nn.softmax(a)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    b_value = sess.run(b)
    print(b_value)
    print(np.e ** x / np.sum(np.e ** x))
