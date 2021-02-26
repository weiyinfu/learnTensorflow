import numpy as np
import tensorflow as tf

N = 3
x_ind = np.arange(N)
np.random.shuffle(x_ind)
y_ind = np.arange(N)
np.random.shuffle(y_ind)

indices = sorted(list(zip(x_ind, y_ind)))
print(indices)
values = np.random.random(N).astype(np.float32)
a = tf.SparseTensor(indices=indices, values=values, dense_shape=(N, N))
a = tf.sparse_tensor_to_dense(a)
# 这句话应该怎么改才对呀？？？？？
aa = tf.sparse_matmul(a, a)
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    a_value = sess.run(a)
    print(a_value)
