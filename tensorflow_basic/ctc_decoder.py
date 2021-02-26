import numpy as np
import pylab as plt
import tensorflow as tf

"""
Connectionist Temporal Classification
tensorflow 提供了两种解码方式：贪心法、beamSearch动态规划法

inputs第一维为时间片
第二维为batchSize
第三维为logits

注意logits的最后一位表示whitespace，什么也不输出


beamSearch不去重似乎不太管用，beamSearch有些难懂
"""
inputs = tf.nn.softmax(tf.constant([[[0.0, 0.9, 0.5],
                                     [0.5, 0.99, 0.2],
                                     [0.9, 0.8, 0.3],
                                     [0.2, 0.9, 0.4],
                                     [0.2, 0.9, 0.4],
                                     ]], dtype=tf.float32), axis=2)
inputs = tf.transpose(inputs, (1, 0, 2))
print('inputs shape', inputs.shape)
decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(inputs, sequence_length=[5], merge_repeated=False)
beam_search_decoded, probabilities = tf.nn.ctc_beam_search_decoder(inputs, sequence_length=[5], top_paths=10, merge_repeated=False)


def getxy(x, y, a):
    return (1 + x) / (a.shape[0] + 2), (1 + y) / (a.shape[1] + 2)


def plt_graph(a):
    for x in range(a.shape[0]):
        for y in range(a.shape[1]):
            xpos, ypos = getxy(x, y, a)
            plt.scatter([xpos], [ypos], s=235, c='r', alpha=0.44)
            plt.text(xpos - 0.02, ypos - 0.02, "%.2f" % a[x, y], size=15, color='g')
    for fx in range(a.shape[0] - 1):
        for fy in range(a.shape[1]):
            for ty in range(a.shape[1]):
                tx = fx + 1
                (fxpos, fypos), (txpos, typos) = getxy(fx, fy, a), getxy(tx, ty, a)
                plt.arrow(fxpos, fypos, txpos - fxpos, typos - fypos, alpha=0.1)


with tf.Session()as sess:
    a = np.squeeze(sess.run(inputs))
    print(a)
    plt_graph(a)
    sess.run(tf.global_variables_initializer())
    print("======")
    print('greedy')
    print(sess.run(decoded))
    print(sess.run(neg_sum_logits))
    print("======")
    print(sess.run(beam_search_decoded))
    print(sess.run(probabilities))
    plt.show()
