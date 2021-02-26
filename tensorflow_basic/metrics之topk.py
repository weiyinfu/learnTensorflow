import numpy as np
import tensorflow as tf

class_count = 10
sampel_count = 10000
labels = tf.constant(np.random.randint(0, class_count, (sampel_count, 1)), dtype=tf.int64)
output = np.random.random((sampel_count, class_count))
logits = tf.constant(output)
order = np.argsort(output, axis=1)[:, ::-1]
my_labels = tf.constant(order)
"""
precision只用于二分类，表示我认为对的里面正确的个数
precision_at_k，precision_at_top_k用于多分类，表示第k个、前k个的正确率

at_k接受的prediction是logits
top_k接受的prediction是class_ids
感觉at_k要比top_k有用得多，top_k基本上想不到使用场景

本程序查看前k名，因为全部都是随机的，所以前k名的precition应该为1/class_count
前k名的recall应该为1/class_count*k

与前k名有关的函数有好多好多：
(sparse+'')*(precision+recall)*(top_k+at_k)=8个函数

* sparse_precision_at_k：已废弃，改为precision_at_k
"""
k = 3
threshold = 0.5

metrics = {
    # 当precision_at_top_k，问题退化为accuracy
    tf.metrics.precision_at_top_k: tf.metrics.precision_at_top_k(labels, my_labels[:, :k]),
    # 经过测试，recall_at_top_k设定k不管用，必须改变label的个数
    tf.metrics.recall_at_top_k: tf.metrics.recall_at_top_k(labels, my_labels[:, :k]),
    tf.metrics.precision_at_k: tf.metrics.precision_at_k(labels, logits, k),
    tf.metrics.recall_at_k: tf.metrics.recall_at_k(labels, logits, k),
}
with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    for func, val in metrics.items():
        print(func.__name__, sess.run(val)[1])
