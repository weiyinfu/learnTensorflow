import numpy as np
import pylab as plt
import tensorflow as tf
from keras.preprocessing import text

"""
skip-gram由缺失词预测上下文:依旧可以使用交叉熵来表示上下文可能的词汇
cbow有上下文预测缺失词
"""
tf.set_random_seed(0)
s = list(map(lambda i: chr(i + ord('a')), range(26))) * 100  # 一个长长的句子
window_size = 2  # 窗口大小
embedding_size = 2  # 嵌入的大小
tok = text.Tokenizer()
tok.fit_on_texts(s)
print(tok.word_index)
num_words = len(tok.word_index) + 1  # 留一个空白

place_x = tf.placeholder(dtype=tf.int32, shape=(None, 1))
place_y = tf.placeholder(dtype=tf.int32, shape=(None, None))
"""
实验证明，embedding初始值关于0对称效果比较好，体现在收敛迅速
"""
embedding = tf.Variable(initial_value=tf.random_uniform((num_words, embedding_size), minval=-1, maxval=1))
embed_y = tf.nn.embedding_lookup(embedding, place_y)
logits = tf.matmul(ctx, tf.matrix_transpose(embedding))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(place_y, (-1,))))
train_op = tf.train.AdamOptimizer(0.1).minimize(loss)


def data_iterator(batch_size=10):
    ind = 0
    batch_x = []
    batch_y = []
    while True:
        if ind + window_size > len(s):
            ind = 0
            continue
        if ind - window_size < 0:
            ind += 1
            continue
        now_x = s[ind - window_size:ind] + s[ind + 1:ind + window_size]
        now_y = s[ind]
        ind += 1
        now_x = [tok.word_index[i] for i in now_x]
        now_y = [tok.word_index[i] for i in now_y]
        batch_x.append(now_x)
        batch_y.append(now_y)
        if len(batch_x) >= batch_size:
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            yield batch_x, batch_y
            batch_x = []  # 清空一下，准备下次使用
            batch_y = []


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data_it = data_iterator()
    for i in range(int(1e8)):
        batch_x, batch_y = next(data_it)
        _, l = sess.run([train_op, loss], feed_dict={
            place_y: batch_y,
            place_x: batch_x,
        })
        if i % 100 == 0:
            print(l)  # loss不可能太小，实验得知loss大概为1.1时就差不多了
        if l < 1.05:
            break
    print('train over')
    embedding_matrix = sess.run(embedding)
    print(embedding_matrix)
    plt.scatter(embedding_matrix[1:, 0], embedding_matrix[1:, 1])
    for x, y, c in zip(embedding_matrix[1:, 0], embedding_matrix[1:, 1] + 0.03, [tok.index_word[i] for i in range(1, num_words)]):
        plt.text(x, y, c)
    plt.show()
