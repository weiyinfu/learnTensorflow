import numpy as np
import tensorflow as tf

"""
nn包下面提供了一些比layers更加小的组合函数

* 运算
* 组合运算
* layer
* 模型级别的，keras的Model，tf的Estimator
"""
w = tf.constant([[1, -2], [-3, 4]])
b = tf.constant([5, 6])
x = tf.constant([[7, 8]])
with tf.Session()as sess:
    print(sess.run(tf.nn.xw_plus_b(x, w, b)))
    print(np.matmul(sess.run(x), sess.run(w)) + sess.run(b))
    print('=' * 10)
    print(sess.run(tf.nn.relu_layer(x, w, b)))
    ans = np.matmul(sess.run(x), sess.run(w)) + sess.run(b)
    ans[ans < 0] = 0
    print(ans)
