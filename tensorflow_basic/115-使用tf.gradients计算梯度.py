import tensorflow as tf

"""
使用tensorflow计算导数
"""
x = tf.Variable(3, dtype=tf.float32)
y = x * x + 4 * x + 3
z = tf.sin(x)
dy_dx = tf.gradients(y, x)
dz_dx = tf.gradients(z, x)
dydz_dx = tf.gradients([y, z], x)  # 等于dy_dx+dz_dx
dy_dxdx = tf.gradients(y, [x, x])  # 分别对xs中的每个元素进行求导
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([dy_dx, dz_dx, dydz_dx, dy_dxdx]))
