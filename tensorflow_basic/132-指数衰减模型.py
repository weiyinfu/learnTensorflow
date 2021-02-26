import matplotlib.pyplot as plt
import tensorflow as tf

"""
在我看来，tensorflow中的许多部件都是花哨的动作，是没有必要存在的

比如global_step这个变量，完全可以交给用户来管理
比如许多变量集合，完全可以用户自己管理，这些集合是tensorflow的默认配置
比如本文的exponential_decay，自动调节学习率（当然也可以用于其它变量），它其实就是一个表达式，学习率还是需要用户自己控制
深度学习真的很简单，一切都是透明的
掌握了optimizer的反向传播原理，就一切都明白了
"""
learn_rate = tf.Variable(1, dtype=tf.float32)
global_step = tf.Variable(0, dtype=tf.float32)
new_rate = tf.train.exponential_decay(learn_rate, global_step=global_step, decay_steps=10, decay_rate=0.2)


def exponential_decay(learn_rate, global_step, decay_steps, decay_rate):
    return learn_rate * decay_rate ** (global_step / decay_steps)


my_rate = exponential_decay(learn_rate=learn_rate, global_step=global_step, decay_steps=10, decay_rate=0.2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = []
    for i in range(10):
        _, rate, old_rate, mine = sess.run([tf.assign(global_step, global_step + 1), new_rate, learn_rate, my_rate])
        a.append(rate)
        print('old rate', old_rate)  # 旧的学习率一直不变
        print('new rate', rate, '== mine', mine)

    plt.plot(a)
    plt.show()
