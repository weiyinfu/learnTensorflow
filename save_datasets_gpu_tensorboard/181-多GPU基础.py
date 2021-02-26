import datetime

import numpy as np
import tensorflow as tf

'''
这个实验分两步：
* 第一步在单个GPU上计算矩阵乘幂
* 第二步在多个GPU上计算矩阵乘幂
如果没有GPU，这个实验是无法做的，所以只看一下代码就可以了。

实验过程:
求A^n+B^n

若为单GPU,则A^n和B^n都在这个GPU上求解
若为双GPU,则A^n和B^n分别在GPU1和GPU2上求解

Basic Multi GPU computation example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
 
This tutorial requires your machine to have 2 GPUs
"/cpu:0": The CPU of your machine.
"/gpu:0": The first GPU of your machine
"/gpu:1": The second GPU of your machine
'''

# Processing Units logs
log_device_placement = True

# Num of multiplications to perform
n = 10  # 计算大矩阵的n次幂

'''
Example: compute A^n + B^n on 2 GPUs
Results on 8 cores with 2 GTX-980:
 * Single GPU computation time: 0:00:11.277449
 * Multi GPU computation time: 0:00:07.131701
'''
# Create random large matrix
A = np.random.rand(10000, 10000).astype('float32')
B = np.random.rand(10000, 10000).astype('float32')

# Create a graph to store results
c1 = []
c2 = []


def matpow(M, n):
    # 递归的方式构建计算图
    if n < 1:  # Abstract cases where n < 1
        return M
    else:
        return tf.matmul(M, matpow(M, n - 1))


'''
Single GPU computing
使用多GPU主要是构图，把不同的操作放在不同的计算组件上
'''
# 最重要的句子上场了:with tf.device('/gpu:0')
with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32, [10000, 10000])
    b = tf.placeholder(tf.float32, [10000, 10000])
    # Compute A^n and B^n and store results in c1
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))

with tf.device('/cpu:0'):
    sum = tf.add_n(c1)  # Addition of all elements in c1, i.e. A^n + B^n

t1_1 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Run the op.
    sess.run(sum, {a: A, b: B})
t2_1 = datetime.datetime.now()

'''
Multi GPU computing
'''
# GPU:0 computes A^n
with tf.device('/gpu:0'):
    # Compute A^n and store result in c2
    a = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(a, n))

# GPU:1 computes B^n
with tf.device('/gpu:1'):
    # Compute B^n and store result in c2
    b = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(b, n))

with tf.device('/cpu:0'):
    sum = tf.add_n(c2)  # Addition of all elements in c2, i.e. A^n + B^n

t1_2 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Run the op.
    sess.run(sum, {a: A, b: B})
t2_2 = datetime.datetime.now()

print("Single GPU computation time: " + str(t2_1 - t1_1))
print("Multi GPU computation time: " + str(t2_2 - t1_2))
