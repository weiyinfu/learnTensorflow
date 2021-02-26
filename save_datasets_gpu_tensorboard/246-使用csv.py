import re

import numpy as np
import tensorflow as tf


def save_csv():
    # Load data
    data = np.load('model/example.npz')
    _x, _y = data["_x"], data["_y"]
    _x = np.concatenate((_x, np.expand_dims(_y, axis=1)), 1)

    # Write to a csv file
    _x_str = np.array_str(_x)
    print("np.array_str", _x_str)
    _x_str = re.sub("[\[\]]", "", _x_str)
    _x_str = re.sub("(?m)^ +", "", _x_str)
    _x_str = re.sub("[ ]+", ",", _x_str)
    with open('model/example.csv', 'w') as fout:
        fout.write(_x_str)


def load_csv():
    # Hyperparams
    batch_size = 10

    # Create a string queue
    fname_q = tf.train.string_input_producer(["model/example.csv"])
    reader = tf.TextLineReader()
    _, value = reader.read(fname_q)

    record_defaults = [[0]] * 10
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = tf.decode_csv(
        value, record_defaults=record_defaults)
    x = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9])
    y = col10

    # Batching
    x_batch, y_batch = tf.train.shuffle_batch(
        [x, y], batch_size=batch_size, capacity=200, min_after_dequeue=100)

    # Ops
    y_hat = 45 - tf.reduce_sum(x_batch, axis=1)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num_epochs * 10):
            _y_hat, _y_batch = sess.run([y_hat, y_batch])
            print(_y_hat, "==", _y_batch)

        coord.request_stop()
        coord.join(threads)


save_csv()
load_csv()
