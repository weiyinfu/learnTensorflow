import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

x = tf.placeholder(dtype=tf.float32, shape=(), name="x")
y = tf.placeholder(dtype=tf.float32, shape=(), name="y")
w = tf.Variable(0.8)
b = tf.Variable(0.2)
yy = tf.sigmoid(w * x + b)
cross = -y * tf.log(yy)
mse = (yy - y) ** 2
cross_grad = tf.gradients(cross, [w, b])
mse_grad = tf.gradients(mse, [w, b])
abs_error = tf.abs(y - yy)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    w_value_list = np.linspace(-8, 8, 100)
    cross_grad_w_list, mse_grad_w_list, abs_error_list = [], [], []
    for w_value in w_value_list:
        sess.run(tf.assign(w, w_value))
        abs_error_value, (cross_grad_w, _), (mse_grad_w, _) = sess.run([abs_error, cross_grad, mse_grad], feed_dict={
            x: 1,
            y: 1
        })
        cross_grad_w_list.append(cross_grad_w)
        mse_grad_w_list.append(mse_grad_w)
        abs_error_list.append(abs_error_value)
    plt.plot(abs_error_list, cross_grad_w_list, label="cross_w=f(A)")
    plt.plot(abs_error_list, mse_grad_w_list, label="mse_w=f(A)")
    plt.xlabel("absolute error")
    plt.ylabel("gradient")
    plt.title("why do we use cross_entropy?")
    plt.legend()
    plt.show()
