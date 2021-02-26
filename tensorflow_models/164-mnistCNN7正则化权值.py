import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

EVAL_INTERVAL_SECS = 10

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1

CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
FC_SIZE = 512

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 600000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "model"
MODEL_NAME = "model.ckpt"


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit


def evaluate(mnist):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None,
                                        IMAGE_SIZE,
                                        IMAGE_SIZE,
                                        1],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        test_images = mnist.validation.images.reshape([-1, 28, 28, 1])
        validate_feed = {x: test_images, y_: mnist.validation.labels}
        y = inference(x, False, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split("/")[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("after %s training steps ,validation accuracy = %g " % (global_step, accuracy_score))
                else:
                    print("no checkpoint file found")
                    return
                time.sleep(EVAL_INTERVAL_SECS)


class l2_regularizer:
    """
    实现tf.contrib.layers.l2_regularizer
    """

    def __init__(self, rate):
        self.rate = rate

    def regularize(self, variable):
        return self.rate * tf.reduce_sum(variable ** 2)

    def __call__(self, variable):
        return self.regularize(variable)


def train(mnist):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    regularizer = l2_regularizer(REGULARIZATION_RATE)
    y = inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
