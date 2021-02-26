import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h))  # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o)  # note that we dont take the softmax at the end because our cost fn does that for us


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = tf.Variable(tf.random_normal(([784, 625]), stddev=0.01))  # create symbolic variables
w_o = tf.Variable(tf.random_normal(([625, 10]), stddev=0.01))

y_mine = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_mine, labels=Y))  # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)  # construct an optimizer
predict_op = tf.argmax(y_mine, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX) + 1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX})))
