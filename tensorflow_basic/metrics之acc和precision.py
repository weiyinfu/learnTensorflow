import tensorflow as tf

# accuracy针对多个类别
_, acc = tf.metrics.accuracy([1, 2, 3, 4], [1, 2, 3, 0])
# precision会把输入cast成bool值，然后计算准确率，准确率=真正对的个数/我认为对的个数
_, precision = tf.metrics.precision([1, 0, 3, 4], [1, 2, 3, 0])
with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    print(sess.run(acc))
    print(sess.run(precision))
