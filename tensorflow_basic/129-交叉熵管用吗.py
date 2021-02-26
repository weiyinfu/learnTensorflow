"""

-sum(p*log(q))为啥能够让q收敛到p

"""
import numpy as np
import tensorflow as tf

sz = 10
q = tf.Variable(tf.random_uniform((sz,)))
pp = np.random.random(sz)
pp = np.e ** pp / np.sum(np.e ** pp)  # 这个地方很重要，如果不进行归一化就无法拟合，归一化不一定非要用softmax
p = tf.constant(pp)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=p, logits=q)
train = tf.train.AdagradOptimizer(0.1).minimize(loss)

with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(400):
        _, l = sess.run([train, loss])
        if i % 100 == 0:
            print(l)
    p_value, q_value = sess.run([p, tf.nn.softmax(q)])
    print(p_value)
    print(q_value)
