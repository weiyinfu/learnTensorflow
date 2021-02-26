"""
ctc_loss：ctc_loss自动会对logits进行softmax，所以我们没有必要执行softmax
"""
import tensorflow as tf
import numpy as np

inputs = tf.nn.softmax(tf.constant([[[0.0, 0.9, 0.5],
                                     [0.5, 0.99, 0.2],
                                     [0.9, 0.8, 0.3],
                                     [0.2, 0.9, 0.4],
                                     [0.2, 0.9, 0.4],
                                     ]], dtype=tf.float32), axis=2)
inputs = tf.transpose(inputs, perm=(1, 0, 2))
labels = tf.SparseTensor(indices=[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)], values=[0, 1, 0, 1, 0], dense_shape=[1, 5])
loss = tf.nn.ctc_loss(labels=labels, inputs=inputs, sequence_length=[5])


def simple_ctc_loss():
    pass


with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    print(np.squeeze(sess.run(inputs)))
    print(sess.run(labels).values)
    print(sess.run(loss))
