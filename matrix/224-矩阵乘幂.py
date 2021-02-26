"""
谁说矩阵乘法就一定是乘法和加法
也有可能是乘幂和乘法


x1^y1*x2^y2=exp(y1*log(x1)+y2*log(x2))

"""
import numpy as np

a = np.random.randint(0, 4, (3, 4))
b = np.random.randint(0, 4, (4, 2))
c = np.ones((a.shape[0], b.shape[1]), dtype=np.float32)
for i in range(a.shape[0]):
    for j in range(b.shape[1]):
        c[i][j] = np.prod(a[i, :].reshape(-1) ** b[:, j].reshape(-1))

print(c)
print(np.exp(np.matmul(np.log(a), b)))
