"""
在执行Optimizer应用梯度、更新变量时，可以指明global_step变量
在两个地方指明：
* minimize()函数中
* apply_gradients()函数中

指明global_step之后,每次执行反向传播都会是global_step++

global_step的作用:
* 调节动量
* 调节学习率
"""
import tensorflow as tf

x = tf.Variable(0, dtype=tf.float32)
global_step = tf.Variable(0, dtype=tf.float32)
op = tf.train.AdamOptimizer()
loss = tf.abs(x)
train = op.minimize(loss, global_step=global_step)

grads = op.compute_gradients(loss)
apply = op.apply_gradients(grads, global_step=global_step)
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        _, step = sess.run([train, global_step])
        print(step)
    print("=" * 10)
    for i in range(10):
        _, step = sess.run([apply, global_step])
        print(step)
    sess.run(tf.assign(global_step, 0))
    print('=' * 10)
    for i in range(10):
        _, step = sess.run([apply, global_step])
        print(step)
