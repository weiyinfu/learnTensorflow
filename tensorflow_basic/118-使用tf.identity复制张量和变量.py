import tensorflow as tf

x = tf.Variable(1.0)
y = tf.Variable(0.0)

# 返回一个op，表示给变量x加1的操作
x_plus_1 = tf.assign_add(x, 1)

"""
如果with语句里面是y=x，那么这个with语句里面没有形成操作结点
所以运行时，根本执行不到x_plus_1操作

当with里面是y=tf.identity时，复制了一个新结点，这个操作就保证了control_dependencies里面的内容会被执行到
"""
with tf.control_dependencies([x_plus_1]):
    # y = x
    y = tf.identity(x)
init = tf.initialize_all_variables()

with tf.Session() as session:
    init.run()
    for i in range(5):
        print(y.eval(), x.eval())  # 相当于sess.run(y)，按
