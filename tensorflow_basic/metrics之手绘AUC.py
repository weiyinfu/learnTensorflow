"""
AUC大合唱

使用tensorflow提供的AUC，sklearn提供的AUC，自定义的AUC，tensorflow metrics库实现的AUC

对于完全随机的数据，最终的AUC必然是0.5，对应的曲线为一条直线
"""
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics

sample_count = 10
y_true = np.random.randint(0, 2, sample_count)
y_mine = np.random.random(sample_count)


# tensorflow的auc貌似有误差
def tf_auc(y_true, y_mine):
    _, acc = tf.metrics.auc(labels=y_true, predictions=y_mine)
    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        return sess.run(acc)


def self_tf_auc(y_true, y_mine):
    # 利用tensorflow提供的基础设施实现AUC
    # AUC只看我认为正确的样本，横轴为我认为正确实际正确，纵轴为我认为正确实际错误
    y_true_tensor = tf.constant(y_true, dtype=tf.float32)
    y_mine_tensor = tf.constant(y_mine, dtype=tf.float32)
    true_count = np.count_nonzero(y_true == 1)
    false_count = np.count_nonzero(y_true == 0)
    # 如果不加上首尾，计算出来的AUC不准
    thresh = np.array([0.0, 1.0] + list(y_mine))
    thresh = thresh[np.argsort(thresh)].astype(np.float32)
    _, false_positive = tf.metrics.false_positives_at_thresholds(y_true_tensor, y_mine_tensor, thresh)
    _, true_positive = tf.metrics.true_positives_at_thresholds(y_true_tensor, y_mine_tensor, thresh)
    true_positive = true_positive / true_count
    false_positive = false_positive / false_count
    with tf.Session()as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        tp, fp = sess.run((true_positive, false_positive))
        return fp, tp


def py_auc(y_true, y_mine):
    index = np.argsort(y_mine)[::-1]
    auc = 0
    negative_count = np.count_nonzero(y_true == 0)
    positive_count = len(y_true) - negative_count
    less_than_me = negative_count
    for ans, mine in zip(y_true[index], y_mine[index]):
        if ans == 1:
            auc += 1 / positive_count * less_than_me / negative_count
        else:
            less_than_me -= 1
    return auc


print(metrics.roc_auc_score(y_true, y_mine))
print(tf_auc(y_true, y_mine))
print(py_auc(y_true, y_mine))
x, y = self_tf_auc(y_true, y_mine)
print(metrics.auc(x, y))
plt.plot(x, y)
plt.xlabel('TruePositive/True')
plt.ylabel('FalsePositive/False')
plt.show()
