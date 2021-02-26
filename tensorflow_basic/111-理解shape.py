import numpy as np
import tensorflow as tf

"""
Tensorflow中有两种shape
一种shape是运行时确定的，另一种shape是构图阶段指明的

tf.shape(tensor)是运行时确定的
tensor.shape是构图阶段指明的
"""


def 使用张量的shape属性():
    a = tf.placeholder(np.float32, [None, 10, 10, 3])  # 10*10的彩色图片
    print(a.get_shape().as_list())
    b = tf.reshape(a, (-1, 300))
    s = tf.shape(a)  # s是一个张量，这个shape是运行时确定的
    """
    a.shape并不是一个tuple，而是TensorShape类型，
    这是一种类似tuple的结构，它的每个元素类型为Dimension类型
    输出（？，10，10，3）
    """
    print(a.shape, type(a.shape))  # 张量的shape属性是构图阶段确定的
    print("TensorShape API")
    print(dir(a.shape))
    print("Dimension API")
    print(dir(a.shape[0]))
    print("==========", type(a.shape[1].value))  # 通过Dimension的value属性获取某一维度的值
    print(b.shape)
    print(np.prod(a.shape[1:]))  # 这种方法非常有用，不需要把维度写死
    c = tf.reduce_sum(a)
    with tf.Session() as sess:
        print(type(tf.shape(a)))  # 这里的shape是一个张量
        print(sess.run([c, s], feed_dict={
            a: np.random.random((1, 10, 10, 3))
        }))


"""
本程序模拟卷积神经网络中卷积层和全连接层之间的连接
"""


def reshape是复制一份吗():
    # 当然不是了
    b = np.random.random(2)  # tf.random是运行时求值，np.random是立即求值
    print(b)
    a = tf.Variable(b)  # 使用np.random()也可以初始化变量
    c = tf.reshape(a, [2, 1])
    d = tf.placeholder(tf.float32)
    print(type(a), type(c), type(d))  # 输出为Variable，Tensor，Tensor
    # 只有Variable类型才可以执行assign操作，Tensor类型无法赋值
    # 执行完reshape之后得到的结果是一个张量，所以无法给reshape之后的数组赋值
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a))
        print(sess.run([tf.assign(a, [4, 5]), a, c]))
        # 这句话会报错，张量没有assign属性，只能给变量赋值
        # print(sess.run([tf.assign(c, [5, 6])]))
        # placeholder也没有assign属性
        # print(sess.run([tf.assign(d, 2)]))
        print("天下大势为我所控")
