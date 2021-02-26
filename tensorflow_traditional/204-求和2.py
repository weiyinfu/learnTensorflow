import tensorflow as tf

n = tf.placeholder(dtype=tf.int32)
s = tf.Variable(0, dtype=tf.int32)


def body(i, s):
    s = s + i
    return i + 1, s


def cond(i, s):
    ans = tf.less(i, n)
    return ans


"""
while_loop中一切皆tensor，body和cond的参数是loop_vars
while_loop的返回值是loop_vars
"""
i, su = tf.while_loop(cond=cond,
                      body=body,
                      loop_vars=[tf.Variable(0), s]
                      )
with tf.control_dependencies([su, s]):
    get_sum = tf.assign(s, su)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(get_sum, feed_dict={
        n: 10
    }))
