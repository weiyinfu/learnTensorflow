"""
激活函数可以用多项式来拟合
这样神经网络就可以写成多项式的形式

用w0*x^0+w1*x^1+w2*x^2=class作为类别,这是一个单层神经网络

上式相当于把x^0,x^1^x2并联起来

如果用在MNIST数据集上则表示这是集成学习,因为在MNIST数据集上要么是0,要么是1,乘幂没有意义

可以对图像进行一些偏移

实践证明,这种方式非常容易发散,误差越来越大,反向传播完全失效
这表明这种训练方式不好,深度学习之所以好是因为分层反馈传播
单层虽然表现力足够强,却很容易发散


"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

IMAGE_SIZE = 28 * 28
LEARN_RATE = 0.1
NUM_CLASS = 10
BATCH_SIZE = 120

power_count = 10
x_place = tf.placeholder(dtype=tf.float32, shape=(None, IMAGE_SIZE))
y_place = tf.placeholder(dtype=tf.float32, shape=(None, NUM_CLASS))
x_transform = tf.map_fn(lambda x: tf.concat([x ** i for i in range(1, power_count + 1)], axis=0), x_place)
print("transformed shape", x_transform.shape)
w = tf.Variable(tf.random_normal((power_count * IMAGE_SIZE, NUM_CLASS,)))
b = tf.Variable(tf.random_normal((NUM_CLASS,)))
logits = tf.matmul(x_transform, w) + b
print(logits.shape, y_place.shape)
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_place)) + 0.1 * tf.norm(w)
train = tf.train.AdamOptimizer(learning_rate=LEARN_RATE).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y_place, axis=1)), dtype=tf.float32))
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100000):
        train_x, train_y = mnist.train.next_batch(BATCH_SIZE)
        train_x -= 0.5
        _, l, acc = sess.run([train, loss, accuracy], feed_dict={
            x_place: train_x,
            y_place: train_y
        })
        if i % 10 == 0:
            print(l, acc)
        if i % 100 == 0:
            test_x, test_y = mnist.test.next_batch(BATCH_SIZE)
            test_x -= 0.5
            acc = sess.run([accuracy], feed_dict={
                x_place: test_x,
                y_place: test_y,
            })
            print("test accuracy", acc)
