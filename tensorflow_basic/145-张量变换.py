import tensorflow as tf
import numpy as np

"""
在使用RNN时，需要进行张量变换
时序是第一维，batch样本是第二维，行是第三维
"""
chunk_size = 1
chunk_n = 2
sample_count = 3
X = tf.Variable(np.arange(sample_count * chunk_n * chunk_size).reshape(sample_count, chunk_n, chunk_size))
transposed = tf.transpose(X, [1, 0, 2])  # 三维矩阵转置，交换前两个轴
reshaped = tf.reshape(transposed, [-1, chunk_size])  # 将数据转化为N*28行，28列的矩阵
reshpe_direct = tf.reshape(transposed, [-1, chunk_n, sample_count, chunk_size])
final = tf.split(reshaped, chunk_n, 0)  # 类似np.split，指定每组的个数，进行分割
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([X, transposed, reshaped, final, reshpe_direct]))

a = np.arange(16)
print(np.split(a.reshape(4, 4), 2, axis=0))
