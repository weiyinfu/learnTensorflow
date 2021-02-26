import tensorflow as tf

"""
pyfunc不能用作训练，只能作为一个过程，一旦假如pyfunc，梯度便再也传不过去了
真尼玛坑
"""
a = tf.Variable(0, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)  # 1不动，0就会来找它，若改为true则二者会向0.5靠近


def f(a, b):
    return abs(a - b)


loss = tf.py_func(f, [a, b], tf.float32)
train = tf.train.AdamOptimizer(0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        _, l = sess.run([train, loss])
        print(l)
    print(sess.run([a, b]))
