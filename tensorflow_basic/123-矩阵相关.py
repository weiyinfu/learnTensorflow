import tensorflow as tf
import numpy as np

"""
有了tensorflow，numpy 甚至都可以少用一点了
tensorflow里面包含了丰富的线性代数运算，从而使得PCA之类的东西可以非常容易地实现
"""
a = tf.fill((2, 3), -1)  # 这是一个张量，而非一个变量
b = tf.zeros_like(a)
c = tf.Variable(a)  # 这是一个变量而非一个张量
d = tf.convert_to_tensor(np.eye(2, 3))
e = tf.diag([1.0, 2.0, 3.0])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([a, b, c]))
    print(sess.run(d))
    print(sess.run(e))
    print(sess.run(tf.matrix_inverse(e)))  # 矩阵求逆
    print(sess.run(tf.matrix_determinant(e)))  # 矩阵行列式值
    print(sess.run(tf.matrix_solve(e, tf.reshape(tf.ones((e.shape[0],)), (-1, 1)))))  # 求解线性方程组
    """
    返回一个元素，第一个是特征值，第二个是特征向量
    """
    print(sess.run(tf.self_adjoint_eig(e)))
    """
    这个rank表示形状
    """
    print(sess.run(tf.rank(e)))
