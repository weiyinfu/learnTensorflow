import tensorflow as tf

"""
自适应学习并不等价于学习率调整，任何optimizer都不会更改学习率，自适应调整调整的是其它东西而不是用户指定的learn_rate，用户指定的learn_rate除非用户自己变动，否则是不会变化的
"""
loss = tf.Variable(3.0, dtype=tf.float32)
learn_rate = tf.Variable(0.01, dtype=tf.float32)
train_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        _, rate, lo = sess.run([train_op, learn_rate, loss])
        print(rate, lo)
