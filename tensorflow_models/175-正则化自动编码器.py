import os.path

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
正则化自动编码器
"""
mnist = input_data.read_data_sets('MNIST_data')

input_dim = 784  # 输入维度
hidden_encoder_dim = 400  # 编码器中间隐层维度
hidden_decoder_dim = 400  # 解码器中间隐层维度
latent_dim = 20  # 隐藏层维度，latent：潜在的，潜伏的
lam = 0.01  # 正则化权重

n_steps = int(1e4)
batch_size = 100


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)


x = tf.placeholder("float", shape=[None, input_dim])
l2_loss = tf.constant(0.0)  # 正则化权重

# 编码器
W_encoder_input_hidden = weight_variable([input_dim, hidden_encoder_dim])
b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)
hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)

# 映射到隐藏层
W_encoder_hidden_mu = weight_variable([hidden_encoder_dim, latent_dim])
b_encoder_hidden_mu = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)
mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim, latent_dim])
b_encoder_hidden_logvar = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

# Sigma encoder
logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

# Sample epsilon
epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

# Sample latent variable
std_encoder = tf.exp(0.5 * logvar_encoder)

"""
z=mu+exp(lamda/2)*epsilon
mu和lamda都是编完码之后的维数为latent的向量
z向量就是最终向量
"""
z = mu_encoder + tf.multiply(std_encoder, epsilon)

W_decoder_z_hidden = weight_variable([latent_dim, hidden_decoder_dim])
b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

# Hidden layer decoder
hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)

W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
b_decoder_hidden_reconstruction = bias_variable([input_dim])
l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)

"""
KLD=-0.5*sum(1+sigma-mu*mu)-exp(sigma)
"""
KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)

x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction
BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x), reduction_indices=1)

loss = tf.reduce_mean(BCE + KLD)

regularized_loss = loss + lam * l2_loss

loss_summ = tf.summary.scalar("lowerbound", loss)
train_step = tf.train.AdamOptimizer(0.01).minimize(regularized_loss)

# add op for merging summary
summary_op = tf.summary.merge_all()

# add Saver ops
saver = tf.train.Saver()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('model', graph=sess.graph)
    # 如果有保存的模型，则加载模型继续训练
    if os.path.isfile("model/model.ckpt"):
        print("Restoring saved parameters")
        saver.restore(sess, "save/model.ckpt")
    else:
        print("Initializing parameters")
        sess.run(tf.global_variables_initializer())

    for step in range(n_steps):
        batch = mnist.train.next_batch(batch_size)
        feed_dict = {x: batch[0]}
        _, cur_loss, summary_str = sess.run([train_step, loss, summary_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

        if step % 50 == 49:
            save_path = saver.save(sess, "model/model.ckpt")
            print("Step {0} | Loss: {1}".format(step, cur_loss))
