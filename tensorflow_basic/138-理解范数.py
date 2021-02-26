import tensorflow as tf
import numpy as np

"""
tf.norm都是针对一维向量进行的范数
"""
N = 3
v = np.random.randint(0, 4, (N, N))
a = tf.Variable(v, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    """
    矩阵范数默认为F范数:弗洛贝尼乌斯范数
    """
    print(sess.run(tf.norm(a)), sess.run(tf.reduce_sum(a * a) ** 0.5))
    # 1范数:列范数
    for i in range(1, 9):
        print(i, '范数', sess.run(tf.norm(a, ord=i)), np.sum(np.abs(v ** i)) ** (1 / i))
    print('无穷范数就是最大值', sess.run(tf.norm(a, ord=np.inf)), np.max(v))  # 无穷范数是行范数
    print('欧几里得范数就是2范数', sess.run(tf.norm(a, ord='euclidean')))  # 欧几里得范数还是2范数
