import tensorflow as tf
from tensorflow.nn.rnn_cell import BasicLSTMCell
from tensorflow.contrib import rnn
from tensorflow.nn import static_rnn
from pprint import pprint

"""
RNN是如何共用权值的?

其实非常简单,只是在时间序列上共用权值
实现上会展成很多层
理论上偏偏要讲的很复杂

RNN必然是只能处理二维数据,而不能够处理三维数据
CNN能够处理RGB图,RNN就不可以
"""
frame_count = 6
frame_size = 2
rnn_hidden_unit = 4
x_place = tf.placeholder(dtype=tf.float32, shape=(None, 6, 2))
x = tf.unstack(x_place, frame_count, axis=1)
print(type(x))
pprint(x)  # x是一个张量列表
lstm_cell = BasicLSTMCell(rnn_hidden_unit)
output, state = static_rnn(lstm_cell, x, dtype=tf.float32)
print(type(output))
pprint(output)
"""
state是一个数组,表示最后一层的c和h,c表示细胞状态,h表示隐状态
"""
print(type(state))
pprint(state)
