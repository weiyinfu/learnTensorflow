import numpy as np
import tensorflow as tf

"""
这个函数怎么用呢？
"""
x = tf.nn.uniform_candidate_sampler(tf.constant(np.arange(20).reshape(-1, 2), dtype=tf.int64), 2, 4, False, 4, seed=0)
print(type(x))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(x))
