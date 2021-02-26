import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf

"""
使用tensorflow提供的函数实现AUC，最后返回一个AUC张量

AUC=the probability of a random chosen positive sample ranks higher 
than a random chosen negative sample.
"""


def body(i, s, y_true, y_mine, true_count, false_count):
    """
    循环体
    :param i: 下标
    :param s: 最终答案
    :param y_true: 真正的标签
    :param y_mine: 我打的标签
    :param true_count: 正样本个数
    :param false_count: 负样本个数
    :return:
    """
    now = tf.cond(tf.equal(y_true[i], 1),
                  true_fn=lambda: tf.divide(
                      tf.divide(tf.count_nonzero(
                          tf.logical_and(tf.less(y_mine, y_mine[i]), tf.equal(y_true, 0)), dtype=tf.float64
                      ) + tf.divide(tf.count_nonzero(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_mine, y_mine[i]))),
                                    2), true_count),
                      false_count),
                  false_fn=lambda: tf.constant(0, dtype=tf.float64))
    return tf.add(i, 1), tf.add(s, now), y_true, y_mine, true_count, false_count


def auc(y_true, y_mine):
    true_count = tf.cast(tf.count_nonzero(tf.equal(y_true, 1)), dtype=tf.float64)
    false_count = tf.cast(tf.count_nonzero(tf.equal(y_true, 0)), dtype=tf.float64)
    s = tf.while_loop(loop_vars=[0, tf.constant(0, dtype=tf.float64), y_true, y_mine, true_count, false_count],
                      cond=lambda i, s, y_true, y_mine, true_count, false_count: tf.less(i, tf.shape(y_mine)[0]),
                      body=body)
    return s[1]


n = 10
y_mine_tensor = tf.placeholder(dtype=tf.float32, shape=(None,))
y_true_tensor = tf.placeholder(dtype=tf.float32, shape=(None,))
auc_node = auc(y_true_tensor, y_mine_tensor)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        y_mine = np.random.random(n)
        y_true = np.random.randint(0, 2, n)
        res = sess.run(auc_node, feed_dict={
            y_mine_tensor: y_mine,
            y_true_tensor: y_true,
        })
        print(res)
        print(metrics.roc_auc_score(y_true, y_mine))
