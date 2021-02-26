import tensorflow as tf

ys = tf.placeholder(dtype=tf.int32, shape=(None,))
max_len = tf.placeholder(dtype=tf.int32, shape=tuple())
length_mask = tf.sequence_mask(ys, maxlen=max_len)  # 长度mask
tensor = tf.constant([1, 2, 3, 4, 5])
boolean_mask = tf.boolean_mask(tensor, [True, False, False, True, False])

in_tensor = tf.placeholder(dtype=tf.int32, shape=(None,))
# how to use sparsemask
# sparse_mask = tf.sparse_mask(in_tensor, ys)
with tf.Session()as sess:
    print(sess.run(length_mask, feed_dict={
        ys: [1, 3],
        max_len: 5,
    }))
    print(sess.run(boolean_mask))
    # print(sess.run(sparse_mask, feed_dict={
    #     ys: [0, 1, 2],
    #     in_tensor: [1, 2, 3, 4, 5, 6],
    # }))
