"""
Dynamic Recurrent Neural Network.
变长序列RNN，比如：音频数据
变长序列RNN是每个batch的长度固定
定长序列RNN是指全部样本的长度固定

实际上本实验还是定长RNN

本实验过程：
* 等概率生成两种序列：随机序列和线性序列（这些序列的长度是不一样的，但是需要补零弄成一样的）
* 让RNN神经网络预测一个序列是线性的还是随机的

TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length. This example is using a toy dataset to classify linear sequences. The generated sequences have variable length.

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

import random

import tensorflow as tf
from tensorflow.nn import static_rnn
from tensorflow.nn.rnn_cell import BasicLSTMCell


# ====================
#  TOY DATA GENERATOR
# ====================
class ToySequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """

    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=1000):
        self.data = []  # data是一个序列列表
        self.labels = []  # 各个样本的标签
        self.seqlen = []  # 各个样本的长度
        for i in range(n_samples):  # 对于每一个样本
            # Random sequence length，当前序列的长度
            len = random.randint(min_seq_len, max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(len)
            # Add a random or linear int sequence (50% prob)
            # 以相同的概率生成线性序列和随机序列
            if random.random() < .5:
                # Generate a linear sequence
                # 生成线性递增序列
                rand_start = random.randint(0, max_value - len)
                s = [[float(i) / max_value] for i in range(rand_start, rand_start + len)]
                # Pad sequence for dimension consistency
                s += [0.] * (max_seq_len - len)
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                # Generate a random sequence
                s = [[float(random.randint(0, max_value)) / max_value]
                     for i in range(len)]
                # Pad sequence for dimension consistency
                s += [0.] * (max_seq_len - len)
                self.data.append(s)
                self.labels.append([0., 1.])
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
seq_max_len = 20  # Sequence max length
n_hidden = 64  # hidden layer num of features，隐藏层神经元个数
n_classes = 2  # linear sequence or not，类别数

# 训练集
trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
# 测试集
testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])  # seqlen也作为一个属性进行输入

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic calculation.
    outputs, states = static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    # 为啥前两个RNN没有这个步骤
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']


pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # Run the initializer
    sess.run(tf.global_variables_initializer())

    for step in range(1, training_steps + 1):
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0 or step == 1:
            # Calculate batch accuracy & loss
            acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            print("Step " + str(step * batch_size),
                  "Minibatch Loss= ", "{:.6f}".format(loss),
                  "Training Accuracy= ", "{:.5f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                        seqlen: test_seqlen}))
