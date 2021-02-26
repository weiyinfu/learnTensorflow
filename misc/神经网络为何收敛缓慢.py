import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
"""
使用sigmoid之后，神经网络之间传递的流量都是0~1之间的数字
使用sigmoid之后，交叉熵虽然可以使得最后一层权重调整幅度正比于绝对误差，但是前面的层却无法做到这一点。这可能是神经网络收敛缓慢的原因。
需要定义出收敛迅速，易于训练的模型。激活函数、损失函数至关重要
改变深度学习的，必定是神经元的重新定义。
"""
x = tf.placeholder(dtype=tf.float32, shape=(), name="x")
y = tf.placeholder(dtype=tf.float32, shape=(), name="y")
w = tf.Variable(0.8)
b = tf.Variable(0.2)
w2 = tf.Variable(0.5)
b2 = tf.Variable(0.2)
activate_function=tf.sigmoid
activate_function=tf.nn.relu
yy = activate_function(w * x + b)
yy = activate_function(w2 * yy + b2)
cross = -y * tf.log(yy)
mse = (yy - y) ** 2
cross_grad = tf.gradients(cross, [w, w2, b])
mse_grad = tf.gradients(mse, [w, w2, b])
abs_error = tf.abs(y - yy)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    w_value_list = np.linspace(-16, 16, 100)
    cross_grad_w_list, mse_grad_w_list, abs_error_list = [], [], []
    for w_value in w_value_list:
        sess.run(tf.assign(w, w_value))
        abs_error_value, (cross_grad_w, cross_grad_w2, _), (mse_grad_w, mse_grad_w2, _) = sess.run([abs_error, cross_grad, mse_grad], feed_dict={
            x: 1,
            y: 1
        })
        cross_grad_w_list.append((cross_grad_w, cross_grad_w2))
        mse_grad_w_list.append((mse_grad_w, mse_grad_w2))
        abs_error_list.append(abs_error_value)
    cross_grad_w_list = np.array(cross_grad_w_list)
    mse_grad_w_list = np.array(mse_grad_w_list)
    plt.plot(abs_error_list, cross_grad_w_list[:, 0], label="cross_w=f(A)")
    plt.plot(abs_error_list, cross_grad_w_list[:, 1], label="cross_w2=f(A)")
    plt.plot(abs_error_list, mse_grad_w_list[:, 0], label="mse_w=f(A)")
    plt.plot(abs_error_list, mse_grad_w_list[:, 1], label="mse_w2=f(A)")
    plt.xlabel("absolute error")
    plt.ylabel("gradient")
    plt.title("why do we use cross_entropy?")
    plt.ylim(-3,3)
    plt.legend()
    plt.show()
