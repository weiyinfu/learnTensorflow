import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
自己动手实现RNN

由此可以发现普通RNN的缺点,收敛缓慢,这是由于梯度消失造成的
"""
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
print(mnist.train.next_batch(3)[0].shape)
x_place = tf.placeholder(dtype=tf.float32, shape=(None, 784))
y_place = tf.placeholder(dtype=tf.float32, shape=(None, 10))
# unstack之后得到一个张量列表
x_transform = tf.unstack(tf.reshape(x_place, (-1, 28, 28)), 28, axis=1)

print(x_transform[0].shape)
rnn_hidden_unit = 100
frame_count = 28
frame_size = 28
# RNN的共享连接
h0 = tf.constant(0, dtype=tf.float32, shape=(rnn_hidden_unit,))
# 权重需要把输入层和上一次的输出并起来
w = tf.Variable(tf.random_normal(shape=(frame_size + rnn_hidden_unit, rnn_hidden_unit)))
b = tf.Variable(tf.random_normal(shape=(rnn_hidden_unit,)))
h = tf.map_fn(lambda x: tf.concat([x, h0], axis=0), x_transform[0])
for i in range(frame_count):
    if i > 0:
        in_tensor = tf.concat([x_transform[i], h], axis=1)
    else:  # 对于0需要特殊处理,因为第一次RNN时上次状态
        in_tensor = h
    h = tf.nn.sigmoid(tf.matmul(in_tensor, w) + b)

fc_w = tf.Variable(tf.random_normal(shape=(rnn_hidden_unit, 10)))
fc_b = tf.Variable(tf.random_normal(shape=(10,)))
logits = tf.matmul(h, fc_w) + fc_b
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_place))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y_place, axis=1)), dtype=tf.float32))
train_op = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100000):
        train_x, train_y = mnist.train.next_batch(batch_size=128)
        _, l, acc = sess.run([train_op, loss, accuracy], feed_dict={
            x_place: train_x,
            y_place: train_y
        })
        if i % 100 == 0:
            print('loss', l, 'acc', acc)
