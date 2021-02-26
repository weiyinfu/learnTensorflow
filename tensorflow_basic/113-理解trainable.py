"""
如果一个变量不可训练，那么这个变量就不会向前传播
"""
import tensorflow as tf

a = tf.Variable(0, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32, trainable=False)  # 1不动，0就会来找它，若改为true则二者会向0.5靠近
loss = tf.abs(b - a)
loss.trainable = False
train = tf.train.AdamOptimizer(0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        _, l = sess.run([train, loss])
        print(l)
    print(sess.run([a, b]))
