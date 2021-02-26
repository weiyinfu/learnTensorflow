import tensorflow as tf
import os

"""
使用simple_save可以迅速搭建一个可用于serving predict的tensorflow模型
实现的是predict的默认函数
"""
tf.Graph().as_default()
x = tf.Variable(0.0)
y = tf.Variable(0.0)
z = x + y
save_dir = "/tmp/simple_save/001"
with  tf.Session() as sess:
    # 如果模型存在就不必要初始化变量了
    sess.run(tf.global_variables_initializer())
    print(sess.run(z, feed_dict={x: 3, y: 4}))
    sess.run(tf.assign_add(x, 2))
    sess.run(tf.assign_add(y, 2))
    print(sess.run(z))
    res = tf.saved_model.simple_save(sess, save_dir, inputs={"x": x, "y": y}, outputs={"z": z})
    print(res)
