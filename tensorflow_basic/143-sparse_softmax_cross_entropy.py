import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

"""
sparse是一种操作的名字，是一个动词，其实就是onehot，更广泛一点就是取值最大的那一维进行onehot
sparse_softmax_cross_entropy_with_logits表示先进行sparse，再进行softmax_cross_entropy_with_logits
sparse_softmax_cross_entropy_with_logits表示不需要使用onehot展开y向量
"""
sz = 3  # 样本数
class_count = 3
# 每个样本的类别
y_true = np.random.randint(0, class_count, sz)
print(y_true)
y_mine = np.random.random((sz, class_count))  # 神经网络的输出：每个样本输出class_count个浮点数
y_one_hot = OneHotEncoder(class_count).fit_transform(y_true.reshape(-1, 1)).toarray()
print(y_one_hot)
y_ = tf.constant(y_true)
y__ = tf.constant(y_one_hot)
y = tf.constant(y_mine)
with tf.Session() as sess:
    print(sess.run(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y)))
    print(sess.run(tf.nn.softmax_cross_entropy_with_logits(labels=y__, logits=y)))
