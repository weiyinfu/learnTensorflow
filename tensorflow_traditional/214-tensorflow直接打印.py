import tensorflow as tf

"""
关于tf.while_loop还有许多不懂之处
* tensorflow中一切都是图，在图中while_loop是如何存在的
* invariant到底是啥意思
* while_loop中数据是如何流动的
"""
n = tf.constant(10)
a = tf.TensorArray(tf.int32, n)


def cond(i, a):
    return i < n


def body(i, a):
    a = a.write(i, tf.Print([i], [i]))
    return i + 1, a


i, a = tf.while_loop(cond, body=body, loop_vars=(0, a))

"""
这个stack非常重要，没有这句话就会报错
"""
a = a.stack()
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(a)
    print(res)
