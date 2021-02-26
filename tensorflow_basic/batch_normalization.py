import numpy as np
import tensorflow as tf

x = np.random.random(10)
scale = 2.0
offset = 1.0
epsilon = 1e-7
x_place = tf.constant(x, dtype=np.float32)
mean, variance = tf.nn.moments(x_place, axes=[0])
y = tf.nn.batch_normalization(x_place, mean=mean, variance=variance, offset=tf.constant(offset, dtype=tf.float32), scale=tf.constant(scale, dtype=tf.float32), variance_epsilon=epsilon)
with tf.Session()as sess:
    print(sess.run(y))
    print((x - np.mean(x)) * scale / (np.var(x) + epsilon) ** 0.5 + offset)
