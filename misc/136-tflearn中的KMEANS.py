"""
Tensorflow也可以用于实现传统机器学习算法！
把传统算法的求解过程使用神经网络来求解，是发论文的一大思路。

本文展示Tensorflow中KMEANS算法在MNIST数据集上的效果
K-Means.最后正确率0.71

原理：
* 聚类（随意k个聚类）
* 给聚类打标签（让聚类中的元素进行投票）
* 为测试集寻找聚类
* 根据测试集所在聚类的标签确定测试用例的类别
* 测试正确率

当聚类个数等于样本数的时候，此算法退化为最近邻

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# 禁用GPU，只需要更改系统环境变量;因为下面的算法并不能充分利用GPU优势
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

epoch_count = 50  # Total steps to train 训练步数
batch_size = 1024  # The number of samples per batch
k = 25  # 聚类的个数
num_classes = 10  # The 10 digits
num_features = 784  # Each image is 28x28 pixels

X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# K-Means Parameters
kmeans = KMeans(inputs=X,  # 输入张量
                num_clusters=k,  # 聚类的个数
                distance_metric='cosine',  # 使用余弦距离
                use_mini_batch=True)

graph = kmeans.training_graph()  # 返回一个张量元组,此函数主要是构图，返回了图中的引脚
all_scores, cluster_idx, scores, cluster_centers_initialized, cluster_center_var, init_op, train_op = graph

cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)  # 计算平均值

sess = tf.Session()

# Run the initializer
sess.run(tf.global_variables_initializer())
sess.run(init_op, feed_dict={X: mnist.train.images})

idx = None
for i in range(epoch_count):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: mnist.train.images})
    if i % 10 == 0:
        print("Step %i, Avg Distance: %f" % (i, d))

# 给每个聚类打标签（通过投票的方式）
# Count total number of labels per centroid, using the label of each training sample to their closest centroid (given by 'idx')
counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]
# Assign the most frequent label to the centroid
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)  # convert_to_tensor很重要

# 虽然刚才已经运行过了，但是下面依旧可以构图、运行
# Evaluation ops
# Lookup: centroid_id -> label
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# Compute accuracy
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test Model
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
