import collections

import tensorflow as tf

# Represent the following array of sets as a sparse tensor:
# a = np.array([[{1, 2}, {3}], [{4}, {5, 6}]])
a = collections.OrderedDict([
    ((0, 0, 0), 1),
    ((0, 0, 1), 2),
    ((0, 1, 0), 3),
    ((1, 0, 0), 4),
    ((1, 1, 0), 5),
    ((1, 1, 1), 6),
])
a = tf.SparseTensor(list(a.keys()), list(a.values()), dense_shape=[2, 2, 2])

# np.array([[{1, 3}, {2}], [{4, 5}, {5, 6, 7, 8}]])
b = collections.OrderedDict([
    ((0, 0, 0), 1),
    ((0, 0, 1), 3),
    ((0, 1, 0), 2),
    ((1, 0, 0), 4),
    ((1, 0, 1), 5),
    ((1, 1, 0), 5),
    ((1, 1, 1), 6),
    ((1, 1, 2), 7),
    ((1, 1, 3), 8),
])
b = tf.SparseTensor(list(b.keys()), list(b.values()), dense_shape=[2, 2, 4])

# `set_difference` is applied to each aligned pair of sets.
with tf.Session() as sess:
    print(sess.run(tf.sets.set_difference(a, b)))
