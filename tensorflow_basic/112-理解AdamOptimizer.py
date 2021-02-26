"""
只有清楚了解各个Optimizer的超参数，才能用好Optimizer
"""
import tensorflow as tf

learn_rate = 1
beta1 = 0.9
beta2 = 0.99
epsilon = 0.01
opitmizer = tf.train.AdamOptimizer(learning_rate=1, beta1=beta1, beta2=beta2, epsilon=epsilon)
x = tf.Variable(10, dtype=tf.float32)
loss = tf.abs(x)
train = opitmizer.minimize(loss)


class MyAdam:
    def __init__(self):
        self.m = 0
        self.v = 0
        self.t = 0

    def minimize(self, g):
        self.t += 1
        lr_t = learn_rate * (1 - beta2 ** self.t) ** 0.5 / (1 - beta1 ** self.t)
        self.m = beta1 * self.m + (1 - beta1) * g
        self.v = beta2 * self.v + (1 - beta2) * g * g
        return -lr_t * self.m / (self.v ** 0.5 + epsilon)


with tf.Session() as  sess:
    sess.run(tf.global_variables_initializer())
    x_mine = sess.run(x)
    ada = MyAdam()
    for i in range(10):
        _, x_value, l = sess.run([train, x, loss])
        x_mine += ada.minimize(l)
        print(x_value, x_mine)
