import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
在hopfield神经网络求解TSP问题中，最重要的是领会损失函数的定义

用全连接求解TSP可能会陷入局部极小解

效果好差呀！

TSP问题使用神经网络求解关键在于损失函数的定义
hopfield神经网络本身就是一个阉割版的神经网络：它只包含了一个激活层

"""
print("import over")

N = 7  # TSP的阶数

city = np.random.random((N, 2))
dis_matrix = np.array([[np.linalg.norm(city[i] - city[j]) for j in range(N)] for i in range(N)])

A = tf.constant(500, tf.float32)
B = tf.constant(500, tf.float32)
C = tf.constant(500, tf.float32)
D = tf.constant(300, tf.float32)
DIS = tf.constant(dis_matrix, tf.float32)

u = tf.Variable(tf.random_uniform((N, N)), dtype=tf.float32)
v = tf.sigmoid(u)

row_loss = tf.constant(0, dtype=tf.float32)
# 每行只有一个1，其余数乘积为0
for x in range(N):
    for i in range(N):
        row_loss += v[x, i] * (tf.reduce_sum(v[x, :]) - v[x, i])
print("so slow")
col_loss = tf.constant(0, dtype=tf.float32)
# 每列只有一个1
for y in range(N):
    for i in range(N):
        col_loss += v[i, y] * (tf.reduce_sum(v[:, y]) - v[i, y])
print("so slow")
# 矩阵之和为N
matrix_loss = tf.reduce_sum(v) - N
# 距离之和最小
dis_loss = tf.constant(0, tf.float32)
for x in range(N):
    for y in range(N):
        we = tf.constant(0, dtype=tf.float32)
        for i in range(N):
            we += v[x, i] * v[y, (i + 1) % N]
        dis_loss += DIS[x, y] * dis_loss
print("so slow")

# loss必须取绝对值，必须保证大于0
row_loss = tf.abs(row_loss)
col_loss = tf.abs(col_loss)
matrix_loss = tf.abs(matrix_loss)
dis_loss = tf.abs(dis_loss)

loss = A * row_loss + B * col_loss + C * matrix_loss + D * dis_loss
train = tf.train.AdamOptimizer(0.1).minimize(loss)


def permutation(a):
    def inner(a, i):
        if i == len(a):
            yield a
        for j in range(i, len(a)):
            a[j], a[i] = a[i], a[j]
            for k in inner(a, i + 1):
                yield k
            a[j], a[i] = a[i], a[j]

    return inner(a, 0)


def get_dis(path):
    return np.sum(dis_matrix[path[i], path[(i + 1) % N]] for i in range(N))


def real_ans():
    """
    暴力求解真正答案
    :return:
    """
    ans = np.sum(dis_matrix)
    best_path = None
    for i in permutation(list(range(N))):
        now = get_dis(i)
        if now < ans:
            ans = now
            best_path = i[:]
    return best_path


def plot_path(path, color, label, sub):
    p = plt.subplot(sub, title=label)
    xs = [city[path[i]][0] for i in range(N)]
    ys = [city[path[i]][1] for i in range(N)]
    p.plot(xs + [city[path[0]][0]], ys + [city[path[0]][1]], linewidth=3, c=color)
    for i in range(len(xs)):
        x, y = (xs[i] + xs[(i + 1) % N]) / 2, (ys[i] + ys[(i + 1) % N]) / 2
        p.text(x, y, "{:.2f}".format(dis_matrix[path[i], path[(i + 1) % N]]), va="center", ha="center")


def valid(row_value, col_value, matrix_value):
    return np.max(row_value) < 0.15 and np.max(col_value) < 0.15 and np.max(matrix_value) < 0.15


# best = real_ans()
# plot_path(best, 'r', 'h', '121')
# plt.show()
# exit(0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    last_loss = 0
    for epoch in range(300000):
        _, l, row_loss_value, col_loss_value, matrix_loss_value = sess.run([train, loss, row_loss, col_loss, matrix_loss])
        if abs(l - last_loss) < 0.1 and valid(row_loss_value, col_loss_value, matrix_loss_value):
            break
        if epoch % 300 == 0:
            last_loss = l
            # 如果若干轮内始终没有变化则停止训练
            print('epoch', epoch, 'loss', l, row_loss_value, col_loss_value, matrix_loss_value)
    ma = sess.run(v)
    print(ma)
    print(np.sum(ma, axis=0), 'row')
    print(np.sum(ma, axis=1), 'col')
    print(np.sum(ma), 'sum')
    temp = np.zeros_like(ma)
    temp[np.argmax(ma, axis=1), np.arange(N)] = 1
    ma = temp
    if not np.all(np.count_nonzero(ma, axis=0) == 1) or not np.all(np.count_nonzero(ma, axis=1) == 1) or not np.count_nonzero(ma) == N:
        raise Exception("invalid solution")
    path = np.argmax(ma, axis=1)
    print(path)
    plt.scatter(city[:, 0], city[:, 1])
    plot_path(path, 'r', 'mine {:.2f}'.format(get_dis(path)), '121')
    ans_path = real_ans()
    plot_path(ans_path, 'b', 'real {:.2f}'.format(get_dis(ans_path)), '122')
    plt.show()
