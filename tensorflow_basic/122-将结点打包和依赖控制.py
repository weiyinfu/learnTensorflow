import tensorflow as tf

"""
Tensorflow中的图管理真是复杂的设计：
结点可以打包,结点打包的目的是为了进行拓扑图的依赖控制
变量可以创建集合


将结点打包，同时执行多个结点

有三个变量a,b,c,先计算他们的和，然后让a,b,c分别increase一个数字，最后在计算一个increase之后的和
"""

tf.reset_default_graph()
a = tf.Variable(1, name="a")
b = tf.Variable(2, name="b")
c = tf.Variable(3, name="c")
with tf.control_dependencies([a, b, c]):
    before_sum = tf.add_n([a, b, c])
with tf.control_dependencies([before_sum]):
    a_add1 = tf.assign(a, tf.add(a, 1, name='a_add1'))
with tf.control_dependencies([before_sum]):
    b_add2 = tf.assign(b, tf.add(b, 2, name='b_add2'))
with tf.control_dependencies([before_sum]):
    c_add3 = tf.assign(c, tf.add(c, 3, name='c_add3'))

with tf.control_dependencies([before_sum]):
    # group操作run之后返回值为None，它只负责同时执行，它并不负责控制依赖
    op = tf.group(a_add1, b_add2, c_add3)
with tf.control_dependencies([op]):  # 如果没有这句话，则sum操作和op操作是并行的，导致出现奇怪的现象
    after_sum = tf.add_n([a, b, c])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([before_sum, op, a, b, c, after_sum]))
"""
执行顺序不同会出现什么情况
"""
