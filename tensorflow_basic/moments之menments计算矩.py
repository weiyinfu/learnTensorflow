import numpy as np
import tensorflow as tf

a = np.random.random((3, 4))
print(a)
mean, variance = tf.nn.moments(tf.constant(a, dtype=np.float32), [1])
with tf.Session()as sess:
    print(sess.run((mean, variance)))
    print(np.mean(a, 1))
    print(np.var(a, 1))
