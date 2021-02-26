""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# 注意这里没用使用onehot，因为使用了Estimator
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)


# Define the neural network
def neural_net(x_dict):
    """
    没有激活函数的神经网络只是线性的，正确率0.91
    线性的处理结果传给下一层线性神经元还是线性的
    使用激活函数正确率0.95
    """
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1, tf.nn.relu)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2, tf.nn.relu)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    """
    在最后输出层可以加上一层softmax，加上之后正确率会略微提升
    softmax层在预测时可以省去，因为它是单纯的增函数，在训练时softmax会影响梯度，有防止过拟合的作用
    softmax可以手动加上，也可以直接通过sparse_softmax_cross_entropy_with_loggits
    """
    pred_probas = tf.nn.softmax(logits)
    pred_classes = tf.argmax(logits, axis=1)

    """
    有两种模式：训练模式，预测模式
    如果是预测模式，忽略softmax层
    """
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=tf.cast(labels, dtype=tf.int32))
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# Tensorflow评估器，需要传入一个函数来指明评估器的一些设置
# Estimator会将传入的数据传给model_f，返回评估结果
model = tf.estimator.Estimator(model_fn)

# Estimator就是一个函数控制器，它调用input_fn来产生数据
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images},
    y=mnist.train.labels,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True)
# 模型训练
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images},
    y=mnist.test.labels,
    batch_size=batch_size,
    shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
