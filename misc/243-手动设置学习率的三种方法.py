"""
如何运行时手动设置学习率？

每个optimizer都拥有
compute_gradients
apply_gradients
两个方法，这两个方法是最重要的方法

grad_and_vars是最重要的结构：它是一个list<pair<grad,var>>


TODO:编写一个自动决定学习率的函数，根据loss序列自动决定学习率


# AttributeError: 'Tensor' object has no attribute 'assign'
# 无法给张量赋值，只能给变量赋值
"""
import tensorflow as tf


def method1():
    """
    optimizer的learnning_rate不仅可以是一个数字，也可以是一个张量
    :return:
    """
    x = tf.Variable(10, dtype=tf.float32)
    learn_rate = tf.placeholder(dtype=tf.float32)
    loss = tf.abs(x)
    train = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            _, x_value, l = sess.run([train, x, loss], feed_dict={learn_rate: 0.1})
            print(x_value, l)


def method2():
    # 直接修改loss值是不允许的，tensor禁止修改
    x = tf.Variable(10, dtype=tf.float32)
    learn_rate = tf.placeholder(dtype=tf.float32)
    loss = tf.abs(x) * learn_rate
    train = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            _, x_value, l = sess.run([train, x, loss], feed_dict={learn_rate: 0.1})
            print(x_value, l)


def method3():
    """
    通过compute_grads然后apply_grads可以控制改变的梯度
    :return:
    """
    x = tf.Variable(10, dtype=tf.float32)
    learn_rate = tf.placeholder(dtype=tf.float32)
    loss = tf.abs(x)
    """
    如果optimizer = tf.train.AdamOptimizer(learning_rate=1)
    则此例不并不生效
    """
    # optimizer = tf.train.AdamOptimizer(learning_rate=1)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
    grads = optimizer.compute_gradients(loss, tf.global_variables())
    # grad是一个list，里面元素是张量和变量列表
    scale_grads = []
    for grad, var in grads:
        scale_grads.append((grad * learn_rate, var))
        # scale_grads.append(tf.assign(grad, grad * learn_rate))
    with tf.control_dependencies([g for g, v in scale_grads]):
        app = optimizer.apply_gradients(scale_grads)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            _, grads_value, x_value, l = sess.run([app, scale_grads, x, loss], feed_dict={learn_rate: 0.1})
            print(grads_value)
            print(x_value, l)


# method1()
# method3()
method2()
