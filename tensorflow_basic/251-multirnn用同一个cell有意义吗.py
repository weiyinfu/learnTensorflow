import numpy as np
import tensorflow as tf

num_units = 3
frame_count = 4
inputs = tf.constant(np.random.random((frame_count, num_units)), dtype=tf.float32)
rnn = tf.nn.rnn_cell.BasicRNNCell(num_units=num_units)
outputs, states = tf.nn.static_rnn(rnn, [inputs], dtype=tf.float32)
layers = 2
rnns = [rnn] * layers
"""
MultiRNNCell中的rnn单元有两种实现方法:
* 层间共用同一个RNN,这跟CNN一样
* 层间不共用同一个RNN,这样模型参数比较多,容量会比较大

两种方式都是有道理的
"""
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnns)
print(rnns[0] == rnns[1])  # 可见这两层是共享权值的
"""
init_state可以是一个张量,可以是place_holder
因为static_rnn的内部使用的BasicRNNCell,所以导致init_state毫无作用
因为BasicRNNCell是没有状态的,只有内部w,b变量
"""
outputs2, states2 = tf.nn.static_rnn(multi_rnn_cell, [inputs], initial_state=tuple(np.zeros((2, frame_count, num_units))))  # 2表示batchsize
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    outputs_value, states_value, outputs_value2, states_value2 = sess.run([outputs, states, outputs2, states2])
    print('single')
    print('output_value')
    print(outputs_value)
    print('state_value')
    print(states_value)
    print('multi')
    print('outputs_value2')
    print(outputs_value2)
    print('states_value2')
    print(states_value2)
