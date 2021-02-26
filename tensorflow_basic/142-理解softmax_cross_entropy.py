import numpy as np
import tensorflow as tf

"""
softmax_cross_entropy_with_logits这个函数常常用来作为loss
它有如下好处：
* 比linalg.norm()速度快

这个函数里面包含两个知识点：
softmax：将向量归一化、非负化到0~1之间
cross_entropy：计算交叉熵，表示两个向量之间的距离

在实际应用中，人们大量使用tf.reduce_mean(tf.softmax_cross_entropy_with_logits())
这种方式求交叉熵。
而实际上，如果label是one-hot的，那么label只有一位为1，
上面这个reduce_mean(softmax_cross_entropy_with_logits)就可以简化为：
log(exp(logits[x])/sum(exp(logits)))
"""
av = np.array([1, 2, 3, 4])
bv = np.array([5, 6, 7, 8])
a = tf.constant(av, dtype=tf.float32)
b = tf.constant(bv, dtype=np.float32)  # np.float32也是可以的

with tf.Session() as sess:
    print(sess.run(tf.nn.softmax_cross_entropy_with_logits(labels=a, logits=b)))
    print(sess.run(-tf.reduce_sum(a * tf.log(tf.nn.softmax(b)))))
    """
    在最后一层根本不需要执行sigmoid，直接下面这个函数
    """
    print(sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=a, logits=b)))


def softmax(a):
    return np.e ** a / np.sum(np.e ** a)


print(-sum(av * np.log(softmax(bv))))
