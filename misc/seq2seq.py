import math
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
使用seq2seq预测时序数据走向
"""


def generate_data(batch_size):
    """
    y=(sin(x+b1),cos(x+b1))
    """
    seq_length = 10

    batch_x = []
    batch_y = []
    noise_size = 0.5
    for _ in range(batch_size):
        rand = random.random() * 2 * math.pi

        sig1 = np.sin(np.linspace(0.0 * math.pi + rand,
                                  3.0 * math.pi + rand, seq_length * 2))
        sig2 = np.cos(np.linspace(0.0 * math.pi + rand,
                                  3.0 * math.pi + rand, seq_length * 2))
        x1 = sig1[:seq_length] + np.random.random(seq_length) * noise_size
        y1 = sig1[seq_length:]
        x2 = sig2[:seq_length] + np.random.random(seq_length) * noise_size
        y2 = sig2[seq_length:]

        x_ = np.array([x1, x2])
        y_ = np.array([y1, y2])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # shape: (batch_size, seq_length, output_dim)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)
    return batch_x, batch_y


sample_x, sample_y = generate_data(batch_size=3)
print("x shape", sample_x.shape)
print("y shape", sample_y.shape)
print("(seq_length, batch_size, output_dim)")

# Internal neural network parameters
# Time series will have the same past and future (to be predicted) lenght.
seq_length = sample_x.shape[0]
batch_size = 5  # Low value used for live demo purposes - 100 and 1000 would be possible too, crank that up!

# Output dimension (e.g.: multiple signals at once, tied in time)
output_dim = input_dim = sample_x.shape[-1]
hidden_dim = 12  # Count of hidden neurons in the recurrent units.
# Number of stacked recurrent cells, on the neural depth axis.
layers_stacked_count = 2

# Optmizer:
learning_rate = 0.007  # Small lr helps not to diverge during training.
# How many times we perform a training step (therefore how many times we
# show a batch).
epoch_count = 150
lr_decay = 0.92  # default: 0.9 . Simulated annealing.
momentum = 0.5  # default: 0.0 . Momentum technique in weights update
lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting

# Definition of the seq2seq neuronal architecture
sess = tf.InteractiveSession()

with tf.variable_scope('Seq2seq'):
    # Encoder: inputs
    # 此处比较有技巧性，有多少个时间片就有多少个input
    encoder_input = [
        tf.placeholder(tf.float32, shape=(
            None, input_dim), name="inpput_{}".format(t))
        for t in range(seq_length)
    ]

    # Decoder: expected outputs
    expected_sparse_output = [
        tf.placeholder(tf.float32, shape=(None, output_dim),
                       name="expected_sparse_output_".format(t))
        for t in range(seq_length)
    ]

    # Give a "GO" token to the decoder.
    # You might want to revise what is the appended value "+ enc_inp[:-1]".
    # decoder的input是encoder_input的最后一个元素和全0
    decoder_input = [tf.zeros_like(encoder_input[0], dtype=np.float32, name="GO")] + encoder_input[:-1]

    # Create a `layers_stacked_count` of stacked RNNs (GRU cells here).
    cells = []
    for i in range(layers_stacked_count):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
            # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    # For reshaping the input and output dimensions of the seq2seq RNN:
    w_in = tf.Variable(tf.random_normal([input_dim, hidden_dim]))
    b_in = tf.Variable(tf.random_normal([hidden_dim], mean=1.0))
    w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
    b_out = tf.Variable(tf.random_normal([output_dim]))

    reshaped_inputs = [tf.nn.relu(tf.matmul(i, w_in) + b_in) for i in encoder_input]

    # Here, the encoder and the decoder uses the same cell, HOWEVER,
    # the weights aren't shared among the encoder and decoder, we have two
    # sets of weights created under the hood according to that function's def.
    # 此处编码器和解码器不是共用cell
    decoder_outputs, decoder_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(
        reshaped_inputs,
        decoder_input,
        cell
    )
    # 输出有一个缩放因子，这一点其实没有必要吧
    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
    # Final outputs: with linear rescaling similar to batch norm,
    # but without the "norm" part of batch normalization hehe.
    reshaped_outputs = [output_scale_factor * (tf.matmul(i, w_out) + b_out) for i in decoder_outputs]

# Training loss and optimizer
with tf.variable_scope('Loss'):
    # L2 loss
    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, expected_sparse_output):
        output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))

    # L2 regularization (to avoid overfitting and to have a  better
    # generalization capacity)
    reg_loss = 0
    for tf_var in tf.trainable_variables():  # 这样写不太好，遍历的变量有点多，不过只遍历一次
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

    loss = output_loss + lambda_l2_reg * reg_loss

with tf.variable_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate, decay=lr_decay, momentum=momentum)
    train_op = optimizer.minimize(loss)


# ## Training of the neural net
def train_batch(batch_size):
    """
    Training step that optimizes the weights
    provided some batch_size X and Y examples from the dataset.
    """
    X, Y = generate_data(batch_size=batch_size)
    feed_dict = {encoder_in: x for encoder_in, x in zip(encoder_input, X)}
    feed_dict.update({output: y for output, y in zip(expected_sparse_output, Y)})
    _, lo = sess.run([train_op, loss], feed_dict)
    return lo


def test_batch(batch_size):
    X, Y = generate_data(batch_size=batch_size)
    feed_dict = {encoder_input[t]: X[t] for t in range(len(encoder_input))}
    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    lo = sess.run([loss], feed_dict)
    return lo


# Training
train_losses = []
test_losses = []

sess.run(tf.global_variables_initializer())
for t in range(epoch_count + 1):
    train_loss = train_batch(batch_size)
    train_losses.append(train_loss)

    if t % 10 == 0:
        # Tester
        test_loss = test_batch(batch_size)
        test_losses.append(test_loss)
        print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t, epoch_count, train_loss, test_loss))

# Plot loss over time:
plt.figure(figsize=(12, 6))
plt.plot(
    np.array(range(0, len(test_losses))) /
    float(len(test_losses) - 1) * (len(train_losses) - 1),
    np.log(test_losses),
    label="Test loss"
)
plt.plot(
    np.log(train_losses),
    label="Train loss"
)
plt.title("Training errors over time (on a logarithmic scale)")
plt.xlabel('Iteration')
plt.ylabel('log(Loss)')
plt.legend(loc='best')
plt.show()

# Test
test_count = 5
print("Let's visualize {} predictions with our signals:".format(test_count))

X, Y = generate_data(batch_size=test_count)
feed_dict = {encoder_input[t]: X[t] for t in range(seq_length)}
outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])

for j in range(test_count):
    plt.figure(figsize=(12, 3))
    for k in range(output_dim):
        past = X[:, j, k]
        expected = Y[:, j, k]
        pred = outputs[:, j, k]

        label1 = "Seen (past) values" if k == 0 else "_nolegend_"
        label2 = "True future values" if k == 0 else "_nolegend_"
        label3 = "Predictions" if k == 0 else "_nolegend_"
        plt.plot(range(len(past)), past, "o--b", label=label1)
        plt.plot(range(len(past), len(expected) + len(past)),
                 expected, "x--b", label=label2)
        plt.plot(range(len(past), len(pred) + len(past)),
                 pred, "o--y", label=label3)

    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()
