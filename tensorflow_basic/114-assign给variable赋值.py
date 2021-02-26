import tensorflow as tf

a = tf.Variable(0, dtype=tf.float32)
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    aa = sess.run(a, feed_dict={  # 此处虽然给a赋值了但是下面并不影响它的值
        a: 3
    })
    print(aa)
    print(sess.run(a))
