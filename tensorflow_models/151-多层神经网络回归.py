import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_function=None):
    # 给定输入神经元个数、输出神经元个数、激活函数，添加一层输入
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            # 使用命名空间可以方便的将结点放到一个地方，方便同时获取这些节点
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 随机初始值的一个矩阵。
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 初始值都是0.1的一个列表.
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases  # 未被激活的值，存在Wx_plus_b中。
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


# 自己定义了所有的数据。
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 建造输入，-1到1区间300个单位【维度】，一个特性有300个例子。
noise = np.random.normal(0, 0.05, x_data.shape)  # 噪声，方差是0.05，格式和x_data一样。
y_data = np.square(x_data) - 0.5 + noise  # 输出y=x^2 -0.5+noise.

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 建立一个隐藏层hidden1（10个神经元），一个输出层prediction（1个神经元）
hidden1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)  # 激活函数选择tf.nn.relu。（可尝试其他激活函数，选择误差结果较小最好）
prediction = add_layer(hidden1, 10, 1, activation_function=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))  # 求一个预测值的平方误差，然后将所有的平方误差求和，最后求和的一个平均值。
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 学习的速率为0.1.

# 这里注意和教程有点差别！！#选定可视化存储目录
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
init = tf.global_variables_initializer()
sess.run(init)
for i in range(100):
    _, lo = sess.run([train_step, loss], feed_dict={
        xs: x_data,
        ys: y_data
    })
    print(i, lo)
x_test = np.linspace(-1, 1, 600)[:, np.newaxis]
y_test = sess.run(prediction, feed_dict={
    xs: x_test
})
plt.scatter(x_data, y_data)
plt.plot(x_test, y_test, color='r', linewidth=5)
plt.show()
