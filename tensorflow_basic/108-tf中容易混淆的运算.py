import tensorflow as tf

"""
tf.multiply和tf.matmul
    multiply相当于numpy中的普通乘法，matmul相当于矩阵乘法

tf.add和tf.nn.bias_add
    tf.nn.bias_add 是 tf.add 中的一个特例
    tf.nn.bias_add支持的功能tf.add都支持
    tf.nn.bias_add 中 bias 一定是 1 维的张量；
    tf.nn.bias_add 中 value 最后一维长度和 bias 的长度一定得一样；
"""


def multiplay_and_matmul():
    with tf.Session() as sess:
        print(sess.run(tf.multiply(3, 4)))
        print(sess.run(tf.multiply(5, [6, 7])))
        print(sess.run(tf.matmul([[1, 2], [3, 4]], [[5], [6]])))
        # Creates a graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(tf.multiply(a, tf.transpose(b))))
    sess.close()


def add_and_bias_add():
    """
    bias_add是add的特例，bias_add支持的功能，add都支持
    :return:
    """
    with tf.Session() as sess:
        print(sess.run(tf.add([3, 4, 5], 4)))
        print(sess.run(tf.nn.bias_add([[1], [2], [3], [4]], [1])))
        print(sess.run(tf.add([[1], [2], [3], [4]], [1])))
        print(sess.run(tf.nn.bias_add([[1, 2, 3],
                                       [4, 5, 6],
                                       [7, 8, 9]],
                                      [1, 2, 3])))
        print(sess.run(tf.add([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]],
                              [1, 2, 3])))
        # 一张2*2的RGB图片，第一个点像素值为[1,2,3]，将它各个颜色通道进行偏移
        print(sess.run(tf.nn.bias_add([[[1, 2, 3], [4, 5, 6]],
                                       [[7, 8, 9], [10, 11, 12]]],
                                      [13, 14, 15])))


def div_and_divide():
    """
    div和divide的区别在于：一个是整数除法，一个是浮点除法
    多么坑爹的标识符
    """
    a = tf.Variable(7, dtype=tf.int32)
    b = tf.Variable(3, dtype=tf.int32)
    with tf.device("cpu:0"):
        with tf.Session()as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run([tf.div(a, b), a // b]))
            print(sess.run([tf.divide(a, b), a / b]))
            print(sess.run([tf.pow(a, b), a ** b]))


add_and_bias_add()
