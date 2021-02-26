import tensorflow as tf
import numpy as np
import random
"""
几乎每个库都会有随机数
Python里面的随机数有三种库可用
* Python语言自带的random包
* numpy中的random包
* tensorflow中的random包
"""
# 在下面的例子中可以发现每次运行都不一样
a = tf.Variable(tf.random_normal((2, 2)))
with tf.Session():
    tf.get_default_session().run(tf.global_variables_initializer())
    print(tf.get_default_session().run(a))
# 使用随机数种子之后，程序的行为变得可控
tf.set_random_seed(0)
a = tf.Variable(tf.random_normal((2, 2)))
with tf.Session():
    tf.get_default_session().run(tf.global_variables_initializer())
    print(tf.get_default_session().run(a))

random.seed(0)
print(random.random())

np.random.seed(0)
print(np.random.random())
