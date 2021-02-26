import numpy as np
import tensorflow as tf
import os


def save_npz():
    x = np.zeros((100, 10), np.int32)
    for i in range(100):
        x[i] = np.random.permutation(10)
    x, y = x[:, :-1], x[:, -1]
    if not os.path.exists("model"): os.mkdir("model")
    np.savez('model/example.npz', x=x, y=y)


def load_npz():
    data = np.load('model/example.npz')
    _x, _y = data["x"], data["y"]
    tf.reset_default_graph()
    x_pl = tf.placeholder(tf.int32, [None, 9])
    y_hat = 45 - tf.reduce_sum(x_pl, axis=1)  # We find a digit x_pl doesn't contain.

    # Session
    with tf.Session() as sess:
        _y_hat = sess.run(y_hat, {x_pl: _x})
        print("y_hat =", _y_hat[:30])
        print("true y =", _y[:30])


tf.reset_default_graph()


def save_tfrecord():
    # Load data
    data = np.load('model/example.npz')
    _x, _y = data["x"], data["y"]

    # Serialize
    with tf.python_io.TFRecordWriter("model/tfrecord") as fout:
        for _xx, _yy in zip(_x, _y):
            ex = tf.train.Example()
            print("tf.train.Example的属性", dir(ex))
            ex.features.feature['x'].int64_list.value.extend(_xx)
            ex.features.feature['y'].int64_list.value.append(_yy)
            s = ex.SerializeToString()
            fout.write(s)


def load_tfrecord():
    fname = "model/example.npz"
    fname_q = tf.train.string_input_producer([fname], num_epochs=1, shuffle=True)
    reader = tf.TFRecordReader()

    # Read the string queue
    _, serialized_example = reader.read(fname_q)
    features = tf.parse_single_example(
        serialized_example,
        features={'x': tf.FixedLenFeature([9], tf.int64),
                  'y': tf.FixedLenFeature([1], tf.int64)}
    )
    x = features['x']
    y = features['y']
    y_hat = 45 - tf.reduce_sum(x)

    # Session
    with tf.Session() as sess:
        # Q5. Initialize local variables
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                _y, _y_hat = sess.run([y, y_hat])
                print(_y[0], "==", _y_hat, end="; ")

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)


def load_tfrecord2():
    tf.reset_default_graph()

    # Load data
    data = np.load('model/example.npz')
    _x, _y = data["x"], data["y"]

    # Hyperparams
    batch_size = 10  # We will feed mini-batches of size 10.
    num_epochs = 2  # We will feed data for two epochs.

    # Convert to tensors
    x = tf.convert_to_tensor(_x)
    y = tf.convert_to_tensor(_y)

    # Q6. Make slice queues
    x_q, y_q = tf.train.slice_input_producer([x, y], num_epochs=num_epochs, shuffle=True)

    # Batching
    x_batch, y_batch = tf.train.batch([x_q, y_q], batch_size=batch_size)

    # Targets
    y_hat = 45 - tf.reduce_sum(x_batch, axis=1)

    # Session
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())

        # Q7. Make a train.Coordinator and threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                _y_hat, _y_batch = sess.run([y_hat, y_batch])
                print(_y_hat, "==", _y_batch)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)


# save_npz()
# load_npz()
save_tfrecord()
load_tfrecord()
