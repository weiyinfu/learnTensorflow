import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

"""
利用秦九韶公式可以拟合任意多项式函数

f(x)=ax^3+bx^2+cx+d
f(x)=((ax+b)x+c)x+d
第一个神经元M，输入a和x，加上b
第二个神经元N，输入M和x，加上c
第三个神经元P，输入N和x，加上d


这种方法似乎具有天然的抗过拟合的性质
但是却很容易陷入局部极小值
"""
n = 20  # 拟合n次函数
data_count = 12

w = tf.Variable(tf.zeros([n]))
x = tf.placeholder(tf.float32, [None, 1], "x")
y = tf.placeholder(tf.float32, [None, 1], "y")
yy = w[0]
for i in range(1, n):
    yy = yy * x + w[i]

# 损失函数使用tf.max很容易震荡，但是对于较小值收敛比较迅速
# 此处的损失函数有多种定义方式
loss = tf.reduce_mean(tf.abs(yy - y))
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

x_from = -1
x_to = 1
x_data = np.linspace(x_from, x_to, data_count)
y_data = np.random.random([data_count])


def f(x, w):
    now = w[0]
    for i in range(1, n):
        now = now * x + w[i]
    return now


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100000000):
        _, lo = sess.run([train_op, loss], feed_dict={
            x: x_data.reshape(-1, 1),
            y: y_data.reshape(-1, 1)
        })
        if epoch % 1000 == 0 or lo < 0.01:
            print(epoch, lo)
            w_value = sess.run(w)
            plt.scatter(x_data, y_data, color='blue')
            x_test = np.linspace(x_from, x_to, 100)
            print(w_value)
            plt.plot(x_test, f(x_test, w_value), color='red', linewidth=3)
            plt.show()
        if lo < 0.01:
            break
