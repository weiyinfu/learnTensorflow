import numpy as np
import tensorflow as tf

"""
对于onehot类型的交叉熵

直接取tf.argmax()下标即可
"""
y_true = tf.constant(np.random.random(10))
y_mine = tf.constant(np.random.random(10))
with tf.Session() as sess:
    print(sess.run(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_mine)))
    print(sess.run(-tf.reduce_sum(y_true * tf.log(tf.nn.softmax(y_mine)))))
