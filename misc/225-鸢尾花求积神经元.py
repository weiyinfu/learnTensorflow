"""
神经元不一定对输入进行加权求和，也可以对输入加权求积
将权值改为幂
将输入加权求和改为输入加权求积
如果有一个为0，就全部为0了

本程序探索各种神经元结构的效率

x1^y1*x2^y2=exp(y1*log(x1)+y2*log(x2))

因为乘幂神经元跟加权求和神经元是等价的，区别只在于对输入进行的激活不太一样

乘幂神经网络与普通神经网络是等价的，区别只在于激活函数不同
对于输入只使用log激活一下，对于每个神经元的输出使用log(softmax(exp(output)))进行激活


实践证明：如果prod(x**w)会导致不收敛，如果sum(x**w)则收敛迅速，但是会有震荡

"""
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

data = load_iris()
data.target = OneHotEncoder().fit_transform(data.target.reshape(-1, 1)).toarray()  # 默认是csr
train_x, test_x, train_y, test_y = train_test_split(data.data, data.target, shuffle=True)
batch_size = train_x.shape[0]

x_place = tf.placeholder(tf.float32, shape=(None, train_x.shape[1]))
y_place = tf.placeholder(tf.float32, shape=(None, train_y.shape[1]))

hidden_units = 100

w = tf.Variable(tf.random_normal((train_x.shape[1], hidden_units), dtype=tf.float32))
b = tf.Variable(tf.random_normal((hidden_units,), dtype=tf.float32))


def wx_b(w, b):
    """
    wx和b应该如何结合起来
    :param w:
    :param b:
    :return:
    """
    return w + b


def activate(s):
    # 激活函数
    return tf.nn.sigmoid(s)


def wx(x, w):
    def wx_one(xi):
        # 可以改动的地方：wx的操作，和对reduce_sum还是reduce_mean还是reduce_prod
        return tf.map_fn(lambda wi: tf.reduce_sum(xi ** wi), tf.transpose(w, [1, 0]))

    return tf.map_fn(wx_one, x)


hidden = activate(wx_b(wx(x_place, w), b))
w2 = tf.Variable(tf.random_normal((hidden_units, train_y.shape[1]), dtype=tf.float32))
b2 = tf.Variable(tf.random_normal((train_y.shape[1],), dtype=tf.float32))
y = wx_b(wx(hidden, w2), b2)
y_mine = tf.argmax(y, 1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_place, logits=y))
accuracy = tf.reduce_mean((tf.cast(tf.equal(y_mine, tf.argmax(y_place, 1)), tf.float32)))
optimizer = tf.train.AdamOptimizer(0.2)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10000):
        _, loss_value, acc, hidden_value, y_value, mine = sess.run([train_op, loss, accuracy, hidden, y, y_mine], feed_dict={
            x_place: train_x,
            y_place: train_y,
        })
        print('epoch', epoch, 'loss', loss_value, 'acc', acc)
        if acc > 0.95:
            break
    acc = sess.run(accuracy, feed_dict={
        x_place: test_x,
        y_place: test_y
    })
    print(acc)
