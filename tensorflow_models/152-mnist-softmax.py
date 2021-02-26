import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
在MNIST数据集上使用softmax
softmax只用来进行训练，而不用来预测
线性方法在MNIST数据集上正确率能够达到92%

使用tensorflow手写逻辑斯地回归
相当于线性回归+softmax，正确率竟然高达0.91

思考：softmax是一个单调递增函数，直接使用argmax就能选择出类别来，为什么非要
凭空加上一个softmax呢？
"""

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次100张照片
batch_size = 100
learning_rate = 0.2
loss_type = "logit"  # 'l2'

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
# square是求平方
# reduce_mean是求平均值
if loss_type == 'l2':
    loss = tf.reduce_mean(tf.square(y - prediction))
elif loss_type == 'logit':
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
else:
    raise Exception("unkonwn loss type")

# 使用梯度下降法来最小化loss，学习率是0.2
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # cast是进行数据格式转换，把布尔型转为float32类型

with tf.Session() as sess:
    # 执行初始化
    sess.run(tf.global_variables_initializer())
    # 迭代21个周期
    for epoch in range(21):
        # 每个周期迭代n_batch个batch，每个batch为100
        for batch in range(mnist.train.num_examples // batch_size):
            # 获得一个batch的数据和标签
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 通过feed喂到模型中进行训练
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        # 计算准确率
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter", epoch, "Accuracy", acc)
