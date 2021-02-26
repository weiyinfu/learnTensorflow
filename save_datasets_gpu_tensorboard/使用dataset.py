import os

import numpy as np
import tensorflow as tf

"""
tensorflow的data体系有两个核心类：
* Dataset
* Iterator

Iterator有如下四种：
单次，
可初始化，
可重新初始化，以及
可馈送。


https://www.tensorflow.org/programmers_guide/datasets


一个注意点：
不要把it.get_next()放在session()内部，尤其不要放在session内部的循环里面，否则最终会导致内存溢出，因为静态图不会回收张量
"""


def text():
    data = tf.data.TextLineDataset([i for i in os.listdir('.') if i.endswith('.py')])
    it = data.repeat().make_one_shot_iterator()
    it_next = it.get_next()
    with tf.Session() as sess:
        for i in range(10):
            print(sess.run(it_next))


def test_one_shot():
    it = tf.data.Dataset.range(0, 10, 2).repeat().make_one_shot_iterator()
    it_next = it.get_next()
    with tf.Session() as sess:
        for i in range(10):
            print(sess.run(it_next))


def random_data():
    dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform((3, 4))).repeat()
    print(dataset.output_classes, dataset.output_shapes, dataset.output_types)
    it = dataset.make_initializable_iterator()
    it_next = it.get_next()
    with tf.Session()as sess:
        sess.run(it.initializer)
        for i in range(10):
            print(sess.run(it_next))


def random_data2():
    dataset = tf.data.Dataset.from_tensor_slices({'x': tf.random_uniform((3, 4)), 'y': tf.random_uniform((3, 2))}).repeat()
    print(dataset.output_classes, dataset.output_shapes, dataset.output_types)
    it = dataset.make_initializable_iterator()
    it_next = it.get_next()
    with tf.Session()as sess:
        sess.run(it.initializer)
        for i in range(10):
            print(sess.run(it_next))


def zip_dataset():
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 2]))
    dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 3])))
    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

    iterator = dataset3.make_initializable_iterator()
    next1, (next2, next3) = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        print(sess.run((next1, next2, next3)))


def iterator_with_init_args():
    """
    可以进行初始化的iterator
    :return:
    """
    max_value = tf.placeholder(tf.int64, shape=[])
    dataset = tf.data.Dataset.range(max_value)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    sess = tf.InteractiveSession()
    # Initialize an iterator over a dataset with 10 elements.
    sess.run(iterator.initializer, feed_dict={max_value: 10})
    for i in range(10):
        value = sess.run(next_element)
        assert i == value

    # Initialize the same iterator over a dataset with 100 elements.
    sess.run(iterator.initializer, feed_dict={max_value: 100})
    for i in range(100):
        value = sess.run(next_element)
        assert i == value


def iterator_with_feeddict():
    """
    iterator模板：一个iterator可以套在多个数据集上
    :return:
    """
    sess = tf.InteractiveSession()
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
    validation_dataset = tf.data.Dataset.range(50)

    # A feedable iterator is defined by a handle placeholder and its structure. We
    # could use the `output_types` and `output_shapes` properties of either
    # `training_dataset` or `validation_dataset` here, because they have
    # identical structure.
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
    next_element = iterator.get_next()

    # You can use feedable iterators with a variety of different kinds of iterator
    # (such as one-shot and initializable iterators).
    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()

    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    # Loop forever, alternating between training and validation.
    while True:
        # Run 200 steps using the training dataset. Note that the training dataset is
        # infinite, and we resume from where we left off in the previous `while` loop
        # iteration.
        for _ in range(200):
            print(sess.run(next_element, feed_dict={handle: training_handle}))

        # Run one pass over the validation dataset.
        sess.run(validation_iterator.initializer)
        for _ in range(50):
            print(sess.run(next_element, feed_dict={handle: validation_handle}))


def multi_head_iterator():
    """
    有多个读头的iterator
    :return:
    """
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
    validation_dataset = tf.data.Dataset.range(50)

    # A feedable iterator is defined by a handle placeholder and its structure. We
    # could use the `output_types` and `output_shapes` properties of either
    # `training_dataset` or `validation_dataset` here, because they have
    # identical structure.
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
    next_element = iterator.get_next()

    # You can use feedable iterators with a variety of different kinds of iterator
    # (such as one-shot and initializable iterators).
    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()

    sess = tf.InteractiveSession()
    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    # Loop forever, alternating between training and validation.
    while True:
        # Run 200 steps using the training dataset. Note that the training dataset is
        # infinite, and we resume from where we left off in the previous `while` loop
        # iteration.
        for _ in range(200):
            sess.run(next_element, feed_dict={handle: training_handle})

        # Run one pass over the validation dataset.
        sess.run(validation_iterator.initializer)
        for _ in range(50):
            sess.run(next_element, feed_dict={handle: validation_handle})


def use_numpy():
    x = np.arange(0, 4)
    # 先shuffle再repeat能够保证遍历完一遍，先repeat再shuffle是对全部数据进行shuffle
    it = tf.data.Dataset.from_tensor_slices((x, x)).shuffle(32).repeat(4).batch(4).make_one_shot_iterator()
    it_next = it.get_next()
    with tf.Session() as sess:
        try:
            while 1:
                print(sess.run(it_next))
                print('=' * 5)
        except tf.errors.OutOfRangeError as err:
            pass


if __name__ == '__main__':
    # test_one_shot()
    # text()
    # random_data()
    # random_data2()
    # zip_dataset()
    # iterator_with_feeddict()
    use_numpy()
