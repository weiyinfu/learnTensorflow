"""

一个矩阵，里面的数字都是[0,1]之间均匀分布。
随机生成的一个方阵A，这个方阵A满秩的概率是多少？

实验证明：tensorflow能够充分利用并行，在求解大型矩阵方面由于CPU
10个4000维线性方程组：
GPU：7.550008058547974
CPU：21.369746208190918

但是在求解小型矩阵上面GPU效果可能不如CPU
"""
import numpy as np
import tensorflow as tf
import time

N = 4000
a = tf.random_uniform((N, N), dtype=np.float32)
b = tf.random_uniform((N, 1), dtype=np.float32)
x = tf.matrix_solve(a, b)
with tf.Session() as sess:
    tf_time = 0
    np_time = 0
    for i in range(10):
        print(i)
        beg = time.time()
        x_value, a_value, b_value = sess.run([x, a, b])
        tf_time += time.time() - beg
        beg = time.time()
        np.linalg.solve(a_value, b_value)
        np_time += time.time() - beg
    print(tf_time, np_time)
