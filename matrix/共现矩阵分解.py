import numpy as np
import pylab as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree


def int2char(a):
    return list(map(lambda i: chr(i + ord('a')), a))


def circle():
    return int2char(range(26)) * 10


def triangle():
    # 为啥不是两个圆圈呢?为啥一边稀疏一边密集？
    s = list(range(13, 25)) * 15 + list(range(13)) * 15
    return int2char(s)


def random():
    return int2char(np.random.randint(0, 25, 1000))


# s = circle()
s = triangle()
# s = random()
print(''.join(s))
window_size = 3  # 窗口大小
dic = dict()
random_index = np.arange(len(set(s)))
np.random.shuffle(random_index)
for i in s:
    if i not in dic:
        dic[i] = random_index[len(dic)]
reverse_dic = dict((v, k) for k, v in dic.items())
word_count = len(dic)
ma = np.zeros((word_count, word_count), dtype=np.float32)
for i in range(len(s)):
    for j in range(max(i - window_size, 0), min(i + window_size, len(s))):
        if i == j: continue
        x, y = dic[s[i]], dic[s[j]]
        ma[x][y] += 1
        ma[y][x] += 1
p = PCA(2, False, True)
res = p.fit_transform(ma)
print(res)
# 归一化之后使用欧式距离，不归一化使用余弦距离
# res /= np.linalg.norm(res, axis=1, keepdims=True)
plt.scatter(res[:, 0], res[:, 1])
for index, (x, y) in enumerate(res):
    plt.text(x, y + 0.05, reverse_dic.get(index))
plt.show()
# 求距离需要使用余弦距离
tr = KDTree(res / np.linalg.norm(res, axis=1, keepdims=True))
while True:
    x = input()
    if x not in dic: continue
    ind = dic[x]
    print("input index", ind)
    dis, neibors = tr.query(res[ind].reshape(1, -1), 10)
    print(neibors)
    print(list(map(lambda i: reverse_dic.get(i), neibors.reshape(-1))))
