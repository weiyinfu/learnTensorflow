import numpy as np
import tensorflow as tf

N = 4
a = tf.Variable(tf.random_uniform((N, N), dtype=np.float32))
"""
svd第一个返回值表示特征根，第二个返回值表示第一个酉阵，
第三个返回值表示第二个酉阵


矩阵特征分解适用于满秩方阵
但是A=VEV^(-1)这个公式只适用于对称矩阵，A=VEV(T)同样只适用于对称矩阵
"""
e, v = tf.self_adjoint_eig(a)
power = tf.assign(a, tf.matmul(a, a))
# 为啥这里不对？loss有点大
mine = tf.matmul(tf.matmul(v, tf.diag(e)), tf.matrix_inverse(v))
loss = tf.norm(mine - a)
vv = tf.matmul(v, tf.matrix_transpose(v))
with tf.Session() as sess:
    sess.run(a.initializer)
    l, vvv, aa, minemine = sess.run([loss, vv, a, mine])
    print(l, '\n\n', vvv, '\n\n', aa, '\n\n', minemine)
