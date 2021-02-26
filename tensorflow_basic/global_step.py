import tensorflow as tf

"""
global_step在tensorflow中似乎占据着极其重要的位置

使用name为global_step即可使得tf.train.get_global_step()获取到此变量
"""
global_step = tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False)
loss = tf.Variable(10.0, dtype=tf.float32)
train_op = tf.train.AdamOptimizer().minimize(loss)
with tf.Session()as sess:
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    print(sess.run(tf.train.get_global_step()))
    sess.run(train_op)
    print(sess.run(tf.train.get_global_step()))  # 依旧输出0
