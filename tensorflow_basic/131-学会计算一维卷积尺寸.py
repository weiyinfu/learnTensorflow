import tensorflow as tf
import numpy as np


def know_conv1d():
    x = tf.placeholder(tf.float32)
    rand = tf.random_normal((2, 2, 1))
    w = tf.Variable(rand)  # 卷积核为2，输入的深度为2，输出的深度为1
    y = tf.nn.conv1d(x, w, 1, padding="SAME")  # stride=1表示每求一次走一步
    with tf.Session() as sess:
        # 每次执行初始化器都会执行初始化操作，如果训练了，会被初始化操作覆盖掉
        rand_val, _ = sess.run([rand, tf.global_variables_initializer()])
        x_input = np.random.rand(1, 5, 2)  # 样本只有1个，这个样本有10个数字，深度为2
        y_output = sess.run(y, feed_dict={x: x_input})
        print(rand_val)
        print(x_input)
        print(y_output.reshape(-1))
        print(np.vdot(rand_val, x_input.flat[:4]))


depth = 1
width = 5
for padding in 'SAME VALID'.split():
    for kernel_size in range(1, width):
        for stride in range(1, width):
            tf.reset_default_graph()
            x_place = tf.placeholder(dtype=tf.float32, shape=(None, 5, depth))
            # w的第一维表示卷积核的大小，第二维表示输入的深度（必须严格等于输入的深度），第三维表示输出的深度
            w = tf.Variable(np.random.randint(0, 4, (kernel_size, depth, 1)), dtype=tf.float32)
            y = tf.nn.conv1d(x_place, w, stride, padding=padding)
            with tf.Session()as sess:
                sess.run(tf.global_variables_initializer())
                x_value = np.ones(x_place.shape[1:])[np.newaxis, :]
                print(x_value.reshape(-1))
                w_value, y_value = sess.run([w, y], feed_dict={
                    x_place: x_value
                })
                print('padding', padding, 'kernel', w_value.reshape(-1), 'stride', stride)
                print(y_value.reshape(-1))
