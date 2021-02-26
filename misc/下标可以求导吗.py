import tensorflow as tf

a = tf.Variable([2, 3, 4, 5], dtype=tf.float32)
ind = tf.Variable(3)
y = a[ind] * a[ind - 1]
g = tf.gradients(y, [a])
print(g)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(g))
