import math

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

mnist = input_data.read_data_sets('MNIST_data')
train_dataset = mnist.train.images
valid_dataset = mnist.validation.images
test_dataset = mnist.test.images
train_labels = mnist.train.labels
valid_labels = mnist.validation.labels
test_labels = mnist.test.labels
print(('Training set', train_dataset.shape, train_labels.shape))
print(('Validation set', valid_dataset.shape, valid_labels.shape))
print(('Test set', test_dataset.shape, test_labels.shape))

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print(('Training set', train_dataset.shape, train_labels.shape))
print(('Validation set', valid_dataset.shape, valid_labels.shape))
print(('Test set', test_dataset.shape, test_labels.shape))

# 创建一个7层网络
layer_sizes = [784, 1000, 500, 250, 250, 250, 10]
L = len(layer_sizes) - 1  # number of layers
num_examples = train_dataset.shape[0]
num_epochs = 100
starter_learning_rate = 0.02
decay_after = 15  # epoch after which to begin learning rate decay
batch_size = 120
num_iter = (num_examples / batch_size) * num_epochs  # number of loop iterations

x = tf.placeholder(tf.float32, shape=(None, layer_sizes[0]))
outputs = tf.placeholder(tf.float32)
testing = tf.placeholder(tf.bool)
learning_rate = tf.Variable(starter_learning_rate, trainable=False)


def bi(inits, size, name):
    return tf.Variable(inits * tf.ones([size]), name=name)


def wi(shape, name):
    return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])


shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))  # shapes of linear layers

weights = {'W': [wi(s, "W") for s in shapes],  # feedforward weights
           # batch normalization parameter to shift the normalized value
           'beta': [bi(0.0, layer_sizes[l + 1], "beta") for l in range(L)],
           # batch normalization parameter to scale the normalized value
           'gamma': [bi(1.0, layer_sizes[l + 1], "beta") for l in range(L)]}

ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
bn_assigns = []  # this list stores the updates to be made to average mean and variance


def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))


# average mean and variance of all layers
running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]
running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]


def update_batch_normalization(batch, l):
    "batch normalize + update average mean and variance of layer l"
    mean, var = tf.nn.moments(batch, axes=[0])
    assign_mean = running_mean[l - 1].assign(mean)
    assign_var = running_var[l - 1].assign(var)
    bn_assigns.append(ewma.apply([running_mean[l - 1], running_var[l - 1]]))
    with tf.control_dependencies([assign_mean, assign_var]):
        return (batch - mean) / tf.sqrt(var + 1e-10)


def eval_batch_norm(batch, l):
    mean = ewma.average(running_mean[l - 1])
    var = ewma.average(running_var[l - 1])
    s = batch_normalization(batch, mean, var)
    return s


def net(x, weights, testing=False):
    d = {'m': {}, 'v': {}, 'h': {}}
    h = x
    for l in range(1, L + 1):
        print("Layer ", l, ": ", layer_sizes[l - 1], " -> ", layer_sizes[l])
        d['h'][l - 1] = h
        s = tf.matmul(d['h'][l - 1], weights['W'][l - 1])
        m, v = tf.nn.moments(s, axes=[0])
        if testing:
            s = eval_batch_norm(s, l)
        else:
            s = update_batch_normalization(s, l)
        s = weights['gamma'][l - 1] * s + weights["beta"][l - 1]
        if l == L:
            # use softmax activation in output layer
            h = tf.nn.softmax(s)
        else:
            h = tf.nn.relu(s)
        d['m'][l] = m
        d['v'][l] = v
    d['h'][l] = h
    return h, d


y, _ = net(x, weights)

cost = -tf.reduce_mean(tf.reduce_sum(outputs * tf.log(y), 1))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1))  # no of correct predictions

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# add the updates of batch normalization statistics to train_step
bn_updates = tf.group(*bn_assigns)
with tf.control_dependencies([train_step]):
    train_step = tf.group(bn_updates)

print("===  Starting Session ===")

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

i_iter = 0
print("=== Training ===")
# print "Initial Accuracy: ", sess.run(accuracy, feed_dict={x: test_dataset, outputs: test_labels, testing: True}), "%"

for i in tqdm(list(range(i_iter, num_iter))):
    # images, labels = mnist.train.next_batch(batch_size)
    start = (i * batch_size) % num_examples
    images = train_dataset[start:start + batch_size, :]
    labels = train_labels[start:start + batch_size, :]
    sess.run(train_step, feed_dict={x: images, outputs: labels})
    if (i > 1) and ((i + 1) % (num_iter / num_epochs) == 0):  # i>1且完成了一个epochs,即所有数据训练完一遍
        epoch_n = i / (num_examples / batch_size)  # 第几个epochs
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_dataset = train_dataset[perm]  # 所有训练数据迭代完一次后，对训练数据进行重排，避免下一次迭代时取的是同样的数据
        train_labels = train_labels[perm]
        if (epoch_n + 1) >= decay_after:
            # decay learning rate
            # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
            ratio = 1.0 * (num_epochs - (epoch_n + 1))  # epoch_n + 1 because learning rate is set for next epoch
            ratio = max(0, ratio / (num_epochs - decay_after))
            sess.run(learning_rate.assign(starter_learning_rate * ratio))
        print("Train Accuracy: ", sess.run(accuracy, feed_dict={x: images, outputs: labels}))

print("Final Accuracy: ", sess.run(accuracy, feed_dict={x: test_dataset, outputs: test_labels, testing: True}), "%")

sess.close()
