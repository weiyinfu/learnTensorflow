import numpy as np
import tensorflow as tf

a = np.array([[1, 2]])
b = np.array([[3], [4]])
print(a @ b)
with tf.Session() as sess:
    x = sess.run(tf.constant(a) @ tf.constant(b))
    print(x)
