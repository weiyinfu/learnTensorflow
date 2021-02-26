import tensorflow as tf

x = tf.placeholder(tf.string, shape=(None,))
y = tf.strings.join([x] * 13)
with tf.Session() as sess:
    print(sess.run(y, feed_dict={
        x: ['one', 'two', 'three']
    }))
    print(sess.run(tf.strings.to_number(x), feed_dict={
        x: '1 2 3 4'.split(),
    }))
