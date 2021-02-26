import tensorflow as tf

"""
操作(op)相当于结点
tensor张量相当于边
整个图相当于一个拓扑图
整个python代码分成两部分：
* 构图
* 运行图

整个构图部分都是在描述图的形状
整个运行图的部分就是tensorflow并发执行多个op的过程

图中有结点op和边tensor
session.run(op),run的是op，点和点之间互相调用，互相调用时，以Tensor的形式传递数据
"""
# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)  # 运行assign会自动执行new_value中的add操作
arr = tf.Variable([0, 1, 2], dtype=tf.float32)
# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.global_variables_initializer()
sayhello = tf.constant("hello world")
# 启动图, 运行 op
with tf.Session() as sess:
    # 运行 'init' op
    sess.run(init_op)
    # 打印 'state' 的初始值
    print(sess.run([sayhello, state]))
    # 运行 op, 更新 'state', 并打印 'state'
    print(sess.run([tf.assign(arr[0], 100), arr]))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
    """
    无法为张量赋值
    AttributeError: 'Tensor' object has no attribute 'assign'
    """
    sess.run(tf.assign(new_value, 35))
