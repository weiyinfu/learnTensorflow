import tensorflow as tf
import numpy as np

n = 100
# 此处不能指明是int32
a = tf.Variable(np.ones(100), trainable=False)

i = tf.constant(2)
j = tf.constant(2)


def set_false(j, i, _):
    update = tf.scatter_nd_update(a, [[j]], [0], name='update_array')
    return tf.add(j, i, name='j_add_i'), i, update


def body(i, j, _):
    # 思考：如何在这个地方直接打印出来呢
    # o = tf.cond(tf.not_equal(a[i], 0), lambda: tf.Print([i], [i]), lambda: 0)
    j = tf.add(i, i)
    inner_loop = tf.while_loop(cond=lambda j, i, _: tf.less(j, n),
                               body=set_false,
                               loop_vars=[j, i, a])
    return tf.add(i, 1, name="outterloop-add1"), inner_loop[0], inner_loop[2]


loop = tf.while_loop(cond=lambda i, j, a: tf.less(i, n),
                     body=body,
                     loop_vars=[i, j, a])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(loop)
    print(res)
    for i in range(2,len(res[2])):
        if res[2][i]:
            print(i, end=' ')
