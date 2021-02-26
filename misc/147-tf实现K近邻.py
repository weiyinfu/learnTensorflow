"""
K近邻算法，正确率高达93%

A nearest neighbor learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

K = 1  # 几近邻？
# In this example, we limit mnist data
Xtrain, Ytrain = mnist.train.next_batch(5000)  # 5000 for training (nn candidates)
Xtest, Ytest = mnist.test.next_batch(200)  # 200 for testing

# tf Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])  # 注意test的维度，这里只能容纳一个测试样本

# Nearest Neighbor calculation using L1 Distance
# 计算差向量的1范数作为两个样板之间的距离
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
# pred = tf.arg_min(distance, 0) #如果是最近邻使用这一句就够了

# topK是一种重要的排序方法
topK = tf.nn.top_k(-distance, K)[1]
topK_labels = tf.nn.embedding_lookup(Ytrain, topK)
class_count = tf.reduce_sum(topK_labels, axis=0)  # 不同类别的个数
pred = tf.arg_max(class_count, 0)
accuracy = 0.

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # loop over test data
    for i in range(len(Xtest)):
        # Get nearest neighbor
        y_mine = sess.run(pred, feed_dict={xtr: Xtrain, xte: Xtest[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", y_mine,
              "True Class:", np.argmax(Ytest[i]))
        # Calculate accuracy
        if y_mine == np.argmax(Ytest[i]):
            accuracy += 1. / len(Xtest)
print("Done!")
print("Accuracy:", accuracy)
