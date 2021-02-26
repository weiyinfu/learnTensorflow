"""
A=BCD
如果C为对角矩阵,DB为单位矩阵,那么A^n就可以快速求出来了

一种神奇的思路:
在矩阵乘幂过程中,为了防止上溢下溢,适时地对矩阵除以一个常数


另一种神奇的思路：
逐次求精
A^n=(B+C)

对于矩阵C，可以直接假设矩阵C是对角矩阵，少了许多变量，编程更简单


这个问题其实是一个求解非线性方程组的问题：

a=bcd

b是N阶方阵，包含N*N个变量
c是对角矩阵，包含N个变量
d是N阶方阵，包含N*N个变量

约束有两个：
bd=e（e为单位阵）
bcd=a

2N*N+N个变量，2N*N个方程

这个非线性方程的解应该是不唯一的
"""
import tensorflow as tf
import numpy as np
import os

N = 3
min_number = -2
max_number = 2
power = 10  # 计算power次幂

a = tf.Variable(min_number + (max_number - min_number) * tf.random_uniform((N, N), dtype=tf.float32))
b = tf.Variable(tf.random_uniform((N, N), dtype=tf.float32))
c = tf.Variable(tf.random_uniform((N,), dtype=tf.float32))  # c是一个对角矩阵
d = tf.Variable(tf.random_uniform((N, N), dtype=tf.float32))
e = tf.eye(N)
bd = tf.matmul(b, d)
bcd = tf.matmul(b * c, d)
A = 1
B = 1
loss = A * tf.reduce_sum((bd - e) ** 2) + B * tf.reduce_sum((bcd - a) ** 2)
train_op = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000000):
        _, l = sess.run([train_op, loss])
        if i % 1000 == 0:
            print(i, l)
            if os.path.exists("over.txt"):
                break
        if l < 1e-8:
            break
        else:
            valid_count = 0
    aa, bb, cc, dd, bcd_value, bd_value = sess.run([a, b, c, d, bcd, bd])
    print('a')
    print(aa)
    print('b')
    print(bb)
    print('c')
    print(cc)
    print('d')
    print(dd)
    print('bcd')
    print(bcd_value)
    print('bd')
    print(bd_value)
    now = np.eye(N)
    for i in range(power):
        now = np.matmul(now, aa)
    mine = np.matmul(bb * cc ** power, dd)
    print('mine')
    print(mine)
    print('power')
    print(now)
