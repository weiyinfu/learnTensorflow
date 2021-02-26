import tensorflow as tf
import numpy as np

"""
以下四个函数只用于2分类
precision
recall
precision_at_threshold
recall_at_threshold

以下函数用于多分类
precision_at_k
precision_at_top_k
"""
y_true = tf.constant([1, 0, 1, 1, 0])
y_mine = tf.constant([1, 0, 0, 0, 1])
logits = tf.constant([0.1, 0.2, 0.5, 0.7, 0.2])
thresh = np.linspace(0, 1, 5, dtype=np.float32)

f = (tf.metrics.precision,
     tf.metrics.recall,
     tf.metrics.accuracy,
     tf.metrics.true_negatives,
     tf.metrics.true_positives,
     tf.metrics.false_negatives,
     tf.metrics.false_positives,
     tf.metrics.precision_at_thresholds,
     tf.metrics.recall_at_thresholds,
     )
values = [val for _, val in (fun(y_true, logits, thresh) if fun.__name__.endswith('thresholds') else fun(y_true, y_mine) for fun in f)]
with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    for fun, val in zip(f, sess.run(values)):
        print(fun.__name__, val)
