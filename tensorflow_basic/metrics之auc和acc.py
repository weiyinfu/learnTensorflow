import tensorflow as tf
import numpy as np

"""
accuracy操作返回两个张量，一个是get张量，一个是set张量，get张量用于获取当前的
"""
# 只要labels和prediction形状相同就可以
old_acc, new_acc = tf.metrics.accuracy(labels=[[1, 2], [3, 4]], predictions=[[1, 2], [1, 4]])
old_auc, new_auc = tf.metrics.auc(labels=np.random.randint(0, 2, 10), predictions=np.random.random(10))
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), ])
    print(sess.run(old_acc))
    print(sess.run(old_auc))
    print(sess.run(new_acc))
    print(sess.run(new_acc))
    print(sess.run(new_auc))
    print(sess.run(new_auc))
    print(sess.run(old_acc))  # 更新之后此值发生变化
    print(sess.run(old_auc))
