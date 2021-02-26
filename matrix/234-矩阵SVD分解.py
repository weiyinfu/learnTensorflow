import numpy as np
import tensorflow as tf

N = 4
a = tf.Variable(tf.random_uniform((N, N), dtype=np.float32))
"""
svd第一个返回值表示特征根，第二个返回值表示第一个酉阵，
第三个返回值表示第二个酉阵
"""
s, v, d = tf.svd(a, compute_uv=True)
power = tf.assign(a, tf.matmul(a, a))
with tf.Session() as sess:
    sess.run(a.initializer)
    aa, ss, vv, dd = sess.run([a, s, v, d])
    a_v = np.matmul(np.matmul(vv, np.diag(ss)), np.asmatrix(dd).transpose())
    print(np.linalg.norm(a_v - aa))
    print(np.matmul(vv, vv.T))  # 这是两个酉阵
    print(np.matmul(dd, dd.T))
    print(vv)
    print(dd)
