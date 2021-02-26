import tensorflow as tf

"""
convert2tensor得到的结果是一堆常量，可以用来查表
"""
a = tf.convert_to_tensor([1, 2, 3], dtype=tf.float32)
print(type(a), a)
loss = tf.reduce_sum(tf.abs(a))
train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    for i in range(10):
        _, lo, aa = sess.run([train_op, loss, a])
        print('loss', lo, 'value', aa)
