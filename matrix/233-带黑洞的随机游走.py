"""
https://www.zhihu.com/question/289014687

一个整型数字x=6000,每次增长101的概率为49.32%，每次减少100元的概率为50.68%，问最终x&gt;7000的概率是多少？
"""
import numpy as np
import tensorflow as tf

lose = 0.4932
win = 1 - lose
win_value = 101
lose_value = 100
init_value = 6000
# 闭区间
ceil_value = 7000
floor_value = 100
A = np.zeros((7102, 7102))
for i in range(A.shape[0]):
    if ceil_value >= i >= floor_value:
        A[i - lose_value, i] = lose
        A[i + win_value, i] = win
    if not ceil_value >= i >= floor_value:
        A[i, i] = 1
A = tf.Variable(A, dtype=np.float32)
assign = tf.assign(A, tf.matmul(A, A))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        sess.run(assign)
        print(i)
    a = sess.run(A)
    p = a[:, init_value].reshape(-1)
    print("overflow", np.sum(p[ceil_value:]), "loop", np.sum(p[floor_value:ceil_value]), "downflow", np.sum(p[:floor_value]))
