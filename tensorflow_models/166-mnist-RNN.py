""" Recurrent Neural Network.

最终精确度高达： 0.992188

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.nn import static_rnn
from tensorflow.nn.rnn_cell import BasicLSTMCell

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.01
training_steps = 5000
batch_size = 128
display_step = 200

num_input = 28  # MNIST data input (img shape: 28*28)
timesteps = 28  # timesteps
num_hidden = 128  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)

X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# 全连接层的权重，这里只有一层全连接

# Prepare data shape to match `rnn` function requirements
# Current data input shape: (batch_size, timesteps, n_input)
# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
# x现在是（样本、时序、行）三维，需要改成（时序、样本、行）三维
# 也就是说在axis=1的方向上拆分，每隔timesteps拆分一个
x = tf.unstack(X, timesteps, axis=1)  # 将一个一维数组拆分成多个数组
print('x type', type(x),x[0].shape)  # x是一个张量列表
# Define a lstm cell with tensorflow
lstm_cell = BasicLSTMCell(num_hidden, forget_bias=1.0)

# Get lstm cell output
# 使用static_rnn,rnn长度不可变
outputs, states = static_rnn(lstm_cell, x, dtype=tf.float32)

print(outputs, states)
# 接上一个全连接
w = tf.Variable(tf.random_normal([num_hidden, num_classes]))
b = tf.Variable(tf.random_normal([num_classes]))
# Linear activation, using rnn inner loop last output
logits = tf.matmul(outputs[-1], w) + b
prediction = tf.nn.softmax(logits)  # 这一步其实没有必要softmax
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
right_count_op = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(training_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy],
                                 feed_dict={X: batch_x,
                                            Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    right_count = 0
    for i in range(0, len(mnist.test.images), batch_size):
        test_data = mnist.test.images[i:i + batch_size].reshape((-1, timesteps, num_input))
        test_label = mnist.test.labels[i:i + batch_size]
        now_right = sess.run(right_count_op, feed_dict={
            X: test_data, Y: test_label
        })
        right_count += now_right
    print("accuracy", right_count / len(mnist.test.images))
