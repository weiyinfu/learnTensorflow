import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_decode
from tensorflow.contrib.crf import viterbi_decode

score = [[
    [1, 2, 3],
    [2, 1, 3],
    [1, 3, 2],
    [3, 2, 1],
]]  # (batch_size, time_step, num_tabs)
transition = [
    [2, 1, 3],
    [1, 3, 2],
    [3, 2, 1],
]  # (num_tabs, num_tabs)
lengths = [len(score[0])]  # (batch_size, time_step)

# numpy
print("[numpy]")
# 可以点进去查看viterbi_decode源码，写的十分精致
np_op = viterbi_decode(
    score=np.array(score[0]),
    transition_params=np.array(transition)
)
print(np_op[0])
print(np_op[1])
print("=============")

# tensorflow
score_t = tf.constant(score, dtype=tf.int32)
transition_t = tf.constant(transition, dtype=tf.int32)
lengths_t = tf.constant(lengths, dtype=tf.int32)
tf_op = crf_decode(
    potentials=score_t,
    transition_params=transition_t,
    sequence_length=lengths_t)
with tf.Session() as sess:
    paths_tf, scores_tf = sess.run(tf_op)
    print("[tensorflow]")
    print(paths_tf)
    print(scores_tf)


def mine(score, transition):
    score = np.squeeze(np.array(score))
    transition = np.squeeze(np.array(transition))
    last = np.empty_like(score)
    score_matrix = np.copy(score)
    last[0] = np.arange(len(last[0]))
    for i in range(1, score.shape[0]):
        for now in range(score.shape[1]):
            for last_state in range(score.shape[1]):
                s = score_matrix[i - 1][last_state] + transition[last_state][now] + score[i][now]
                if s > score_matrix[i][now]:
                    score_matrix[i][now] = s
                    last[i][now] = last_state
    now = np.argmax(score_matrix[-1])
    path = [now]
    for i in range(len(last) - 1, 0, -1):
        path.append(last[i][now])
        now = last[i][now]
    return np.max(score_matrix[-1]), path[::-1]


print("=" * 10)
print('[mine]')
score_mine, path_mine = mine(score, transition)
print(path_mine)
print(score_mine)
