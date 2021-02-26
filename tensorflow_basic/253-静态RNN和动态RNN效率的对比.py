import time

import numpy as np
import pylab as plt
import tensorflow as tf
from keras.datasets import mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data('/data/mnist.npz')
sz = 28
category_count = 10
rnn_units = 32


class DataIterator:
    def __init__(self, batch_size):
        self.index = np.arange(len(train_x))
        np.random.shuffle(self.index)
        self.now = 0
        self.batch_size = batch_size

    def get_batch(self):
        if self.now + self.batch_size > len(self.index):
            np.random.shuffle(self.index)
            self.now = 0
        ind = self.index[self.now:self.now + self.batch_size]
        self.now += self.batch_size
        return train_x[ind], train_y[ind]


def use_static_rnn(x_place):
    x_transposed = tf.transpose(x_place, (1, 0, 2))
    x_sequence = tf.unstack(x_transposed)
    rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_units)
    initial_state = rnn_cell.zero_state(tf.shape(x_place)[0], dtype=tf.float32)
    rnn_out, states = tf.nn.static_rnn(rnn_cell, x_sequence, initial_state=initial_state)
    return rnn_out[-1]


def use_dynamic_rnn(x_place):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_units)
    init_state = cell.zero_state(tf.shape(x_place)[0], dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell, x_place, initial_state=init_state)
    return outputs[:, -1, :]


def main(rnn):
    tf.reset_default_graph()
    x_place = tf.placeholder(dtype=tf.float32, shape=(None, sz, sz))
    y_place = tf.placeholder(dtype=tf.int32, shape=(None))
    rnn_out = rnn(x_place)
    y = tf.reshape(rnn_out, (-1, rnn_units))
    w = tf.Variable(tf.random_normal((rnn_units, category_count)))
    b = tf.Variable(tf.random_normal((category_count,)))
    logits = tf.matmul(y, w) + b
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_place))

    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
    mine = tf.argmax(logits, axis=1, output_type=tf.int32)
    right_or_wrong = tf.cast(tf.equal(mine, y_place), tf.float32)
    accuracy = tf.reduce_mean(right_or_wrong, axis=0)
    right_count = tf.reduce_sum(right_or_wrong, axis=0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data_it = DataIterator(128)
        beg_time = time.time()
        acc_series = []
        step_series = []
        for i in range(int(1e8)):
            batch_x, batch_y = data_it.get_batch()
            _, lo, acc = sess.run([train_op, loss, accuracy], feed_dict={
                x_place: batch_x,
                y_place: batch_y,
            })
            print('method', rnn.__name__, 'global step', i, 'loss', lo, 'accuracy', acc, 'time used', time.time() - beg_time)
            acc_series.append((acc, time.time() - beg_time))
            step_series.append((i, time.time() - beg_time))
            if acc > 0.90:
                break
        batch_size = 128
        s = 0
        for i in range(0, len(test_y), batch_size):
            batch_x, batch_y = test_x[i:i + batch_size], test_y[i:i + batch_size]
            cnt, acc = sess.run([right_count, accuracy], feed_dict={
                x_place: batch_x,
                y_place: batch_y,
            })
            s += cnt
        print('test accuracy', s / len(test_y))
        acc_series = np.array(acc_series)
        acc_series[:, 1] -= acc_series[0, 1]  # 时间以第一次训练结束时间为准
        step_series = np.array(step_series)
        step_series[:, 1] -= step_series[0, 1]
        return acc_series, step_series


acc_series1, step_series1 = main(use_static_rnn)
acc_series2, step_series2 = main(use_dynamic_rnn)
fig, (one, two, three) = plt.subplots(1, 3)
one.set_title('time-accuracy')
one.plot(acc_series1[:, 1], acc_series1[:, 0], label='static')
one.plot(acc_series2[:, 1], acc_series2[:, 0], label='dynamic')
one.set_xlabel('time')
one.set_ylabel('accuracy')
one.legend()
two.set_title('time-step')
two.plot(step_series1[:, 1], step_series1[:, 0], label='static')
two.plot(step_series2[:, 1], step_series2[:, 0], label='dynamic')
two.set_xlabel('time')
two.set_ylabel("step")
two.legend()
three.set_title('step-accuracy')
three.set_xlabel('step')
three.set_ylabel('accuracy')
three.plot(step_series1[:, 0], acc_series1[:, 0], label='static')
three.plot(step_series2[:, 0], acc_series2[:, 0], label='static')
three.legend()
plt.show()
