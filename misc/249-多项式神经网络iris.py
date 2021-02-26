"""
iris数据集每个样本记为X
用w0*x^0+w1*x^1+w2*x^2=class作为类别,这是一个单层神经网络

上式相当于把x^0,x^1^x2并联起来
"""
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

data = load_iris()
y = OneHotEncoder().fit_transform(data.target.reshape(-1, 1)).toarray()
train_x, test_x, train_y, test_y = train_test_split(data.data, y)
print(train_x.shape, test_x.shape)

power_count = 10
x_place = tf.placeholder(dtype=tf.float32, shape=(None, train_x.shape[1]))
y_place = tf.placeholder(dtype=tf.float32, shape=(None, train_y.shape[1]))
x_transform = tf.map_fn(lambda x: tf.concat([x ** i for i in range(1, power_count + 1)], axis=0), x_place)
print("transformed shape", x_transform.shape)
w = tf.Variable(tf.random_normal((power_count * train_x.shape[1], train_y.shape[1],)))
b = tf.Variable(tf.random_normal((train_y.shape[1],)))
logits = tf.matmul(x_transform, w) + b
print(logits.shape, y_place.shape)
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_place))
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y_place, axis=1)), dtype=tf.float32))
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _, l, acc = sess.run([train, loss, accuracy], feed_dict={
            x_place: train_x,
            y_place: train_y
        })
        if i % 10 == 0:
            print(l, acc)
        if i % 100 == 0:
            acc = sess.run([accuracy], feed_dict={
                x_place: test_x,
                y_place: test_y,
            })
            print("test accuracy", acc)
