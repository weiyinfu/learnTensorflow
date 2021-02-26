import tensorflow as tf

"""
输入一个年龄，如果小于18岁，打印young
否则打印old
"""


def f1():
    o = tf.constant("old")
    return tf.Print(input_=o, data=[o])


def f2():
    o = tf.constant("young")
    return tf.Print(input_=o, data=[o])


age = tf.placeholder(tf.int32)

r = tf.cond(tf.greater(age, 18), f1, f2)
with tf.Session() as sess:
    print(str(sess.run(r, feed_dict={age: 19}), 'gbk'))
