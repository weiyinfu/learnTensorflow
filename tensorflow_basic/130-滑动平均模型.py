"""
滑动平均模型
ExponentialMovingAverage相当于代理
对数据的写操作都是作用在真实值上,对数据的读操作都通过代理

x=k*lastValue+(1-k)*nowValue,此处的k称为"保守程度"
随着训练轮数的增加,保守程度逐渐变大,它逐渐趋近于1,就像老人一样,年龄大了,
学习能力就弱了;人见多识广之后,就会产生根深蒂固的思想,不会轻易相信外界传
进来的数据.

调用形式:
tf.train.ExponentialMovingAverage(decay=0.99, num_updates=step)
decay的值是趋近于1的,一旦趋近于1,就会丧失学习能力.所以需要给decay确定一个上界
这种机制其实就是一种防止过拟合,尽量保持曲线的平滑,尽量保持模型的沉稳,不至于因为外物的一丝扰动就变得过于激动.

如果不指定num_updates参数,则k值为一常量
当提供num_updates参数的时候,使用k=min(decay,(1+step)/(10+step))
"""

import numpy as np
import tensorflow as tf

a = tf.Variable(0, dtype=tf.float32)
global_step = tf.Variable(0, trainable=False)
ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step)
op = ema.apply([a])  # ema只能初始化一次
# op = ema.apply([a])  # ema只能初始化一次 Moving average already computed for: Variable:0
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 未经op就average不管用,所以下面两次打印结果相同
print(sess.run([a, ema.average(a)]))  # 0，0
sess.run(tf.assign(a, 1))  # a的值发生改变，即便a发生了改变，但是并没有通知ema
print(sess.run([a, ema.average(a)]))

sess.run(op)  # 执行op操作，才能通知ema
decay = 1 / 10
a1 = np.dot((decay, 1 - decay), (0, 1))
print(sess.run([a, ema.average(a), global_step]), a1)

# 下面讲解一下原理
new_step = 15  # 假设已经训练了15轮了
decay = min(0.99, (1 + new_step) / (10 + new_step))
a2 = np.dot((decay, 1 - decay), (a1, 1))
sess.run(tf.assign(global_step, new_step))  # step+=1
sess.run(op)  # 多运行一次还会继续赋值
print(sess.run([a, ema.average(a), global_step]), a2)
print(sess.run([a, ema.average(a), global_step]), a2)  # 执行多次并不改变
print(sess.run([a, ema.average(a), global_step]), a2)  # 执行多次并不改变
sess.close()
"""
EMA是一个管理器
执行apply操作表示向EMA管理器输入数据
执行average操作表示从EMA读取数据
"""
