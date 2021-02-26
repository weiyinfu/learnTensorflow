import tensorflow as tf

n = tf.placeholder(dtype=tf.int32)


def body(i, s):
    """
    此函数只在构图时调用一次
    这表明循环确实是循环
    :param i:
    :param s:
    :return:
    """
    s += i
    print(type(s))
    return i + 1, s


def cond(i, s):
    return tf.less(i, n)


"""
while_loop中一切皆tensor，body和cond的参数是loop_vars
while_loop的返回值是loop_vars
"""
loop = tf.while_loop(cond=cond,
                     body=body,
                     # 虽然赋值的时候是int，但是会被转换成Tensor
                     loop_vars=[0, 0]
                     )
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    res = sess.run(loop, feed_dict={
        n: 10
    })
print(res)
