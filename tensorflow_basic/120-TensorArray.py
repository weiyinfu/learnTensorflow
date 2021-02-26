import tensorflow as tf

"""
TensorArray中最重要的操作就是stack和unstack
stack把一系列张量堆叠成一个张量

TensorArray跟tf.while是一对好朋友
"""


def condition(time, output_ta_l):
    return tf.less(time, 3)


def body(time, output_ta_l):
    output_ta_l = output_ta_l.write(time, [2.4, 3.5])
    return time + 1, output_ta_l


time = tf.constant(0)
output_ta = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

last_time, last_out = tf.while_loop(condition, body, loop_vars=[time, output_ta])

final_out = last_out.stack()

with tf.Session():
    print(last_time.eval())
    print(final_out.eval())
