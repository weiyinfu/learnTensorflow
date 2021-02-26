"""
Wx+b=y求W和b
用神经网络BP算法求W和b
"""
import numpy as np
import tensorflow as tf

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.random.rand(2, 100).astype(np.float32)  # 随机输入
y_data = np.dot([0.1, 0.2], x_data) + 0.3  # 使用点乘生成y

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))  # 一开始的偏移量
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))  # 一开始的权重
y = tf.matmul(W, x_data) + b

# 最小化方差:loss,optimizer,train这三大元素一旦确定，反向求导的过程就确定了
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
print(sess.run([W, b]))
sess.close()