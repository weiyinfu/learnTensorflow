import numpy as np
import tensorflow as tf

tf.set_random_seed(0)
inputs = tf.placeholder(dtype=tf.float32, shape=(None, None, 3))
rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=3)
init_state = rnn_cell.zero_state(tf.shape(inputs)[0], dtype=tf.float32)
outputs, states = tf.nn.dynamic_rnn(rnn_cell, inputs, initial_state=init_state)
print(init_state)
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    init_state_value = sess.run(init_state, feed_dict={
        inputs: [np.arange(6).reshape(2, 3)]
    })
    print('first ,let\'s view the init_state_value')
    print(init_state_value)

    print('second,let\'s feed the RNN with all the frames')
    outputs_value, states_value = sess.run([outputs, states], feed_dict={
        inputs: [np.arange(6).reshape(2, 3)]
    })
    print(outputs_value)
    print(states_value)

    print('third,let\'s feed the RNN one by one,the result should be as same as above')
    _, first_states = sess.run([outputs, states], feed_dict={
        inputs: [[[0, 1, 2]]]
    })
    print(first_states)
    outputs, _ = sess.run([outputs, states], feed_dict={
        inputs: [[[3, 4, 5]]],
        init_state: first_states
    })
    print(outputs)
