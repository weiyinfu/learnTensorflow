import tensorflow as tf
import numpy as np

"""
变量初始化只能初始化一次

tf.random_normal()是一个操作，这个操作没有输入，只有输出，输出是一个张量

当使用tf.random_normal()输出的张量初始化tf.Variable时，Variable的值就发生变化了

不同的Variable有不同的initializer，这个initializer相当于一个tf.assign()操作
"""
a = tf.random_normal((1,))  # a的值每次都会变化
print(a)
b = tf.add(a, a)
c = tf.Variable(a)
d = tf.Variable(a)
with tf.Session() as sess:
    a_ar, _ = sess.run([a, tf.global_variables_initializer()])  # 此时c的值已经确定了
    print(a_ar)
    print(np.array(sess.run([a, b, c, d])).reshape(-1))  # c的值不变
    sess.run(tf.global_variables_initializer())  # 再次运行初始化，c的值发生改变
    print(np.array(sess.run([a, b, c, d])).reshape(-1))
    print(np.array(sess.run([a, b, c, d])).reshape(-1))
    sess.run(c.initializer)  # 也可以单独运行某个variable的initializer，不会影响其他数据(如d)
    print(np.array(sess.run([a, b, c, d])).reshape(-1))
