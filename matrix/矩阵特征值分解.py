"""
理解矩阵特征值分解和SVD分解的区别

对于对称矩阵来说，SVD和特征值分解是等价的
对于普通矩阵则不然
"""
import numpy as np

n = 3
a = np.random.random((n, n))
# 使a变成对称矩阵
a = a + a.T
root, vec = np.linalg.eig(a)
print(root)
print(vec)
s, v, d = np.linalg.svd(a)
print(s)
print(v)
print(d)
