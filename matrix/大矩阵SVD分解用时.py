import time

import numpy as np
import tensorflow as tf

"""
GPU dtype=tf.float64竟然比CPU还要慢,改成tf.float32就快了
大约是O(n**3)的复杂度，所以倍数很容易求出来

GPU的一些数据（用CPU已经完全行不通了）
1000~=8s
3000~=27s
4000~=54s
5000~=112s

规律大约是用时越来越多
这说明使用矩阵分解的方式实现小数据集的word2vec高效而精确（必然比gensim和fasttext快）
"""
n = 3
a = np.random.random((n, n))


def use_numpy():
    return np.linalg.svd(a)


def use_tf():
    v = tf.Variable(a, dtype=tf.float32)
    svd = tf.svd(v)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(svd)
        return res


print(use_numpy())
print(use_tf())
now = time.time()
# use_numpy()
# print(time.time() - now)
# now = time.time()
# use_tf()
# print(time.time() - now)
