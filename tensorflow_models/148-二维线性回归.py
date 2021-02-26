import matplotlib.pyplot as plt
import numpy  as np
import tensorflow as tf

"""
可视化线性回归
"""
learning_rate = 0.01
training_epochs = 1000
display_step = 50
N = 50
# Training Data
train_x = np.random.random(N) * 5
w = np.random.random() * 3
b = np.random.random() * 4
train_y = w * train_x + b + np.random.normal(0, 0.2, N)
print("real w", w, 'real b', b)
X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

pred = tf.add(tf.multiply(W, X), b)

# 方差
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * N)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for (x, y) in zip(train_x, train_y):
            sess.run(train, feed_dict={X: x, Y: y})

        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_x, Y: train_y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
