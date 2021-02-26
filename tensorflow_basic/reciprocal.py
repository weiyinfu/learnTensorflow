import tensorflow as tf

# reciprocal就是取倒数
x = tf.constant([1, 2, 3.0])
with tf.Session()as sess:
    print(sess.run(tf.reciprocal(x, )))
