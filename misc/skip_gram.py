import collections
import math

import numpy as np
import tensorflow as tf


class Word2Vec:
    def __init__(self,
                 vocab_list,  # 词的个数
                 embedding_size,  # 词向量的长度
                 window_size,  # 单边窗口长
                 num_sampled,  # 负采样个数
                 learning_rate,
                 ):
        self.vocab_list = vocab_list
        self.vocab_size = vocab_list.__len__()
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        # word => id 的映射
        self.word2id = dict(zip(self.vocab_list, range(self.vocab_size)))

        # 首先定义两个用作输入的占位符，分别输入输入集(train_inputs)和标签集(train_labels)
        self.place_x = tf.placeholder(tf.int32, shape=[None])
        self.place_y = tf.placeholder(tf.int32, shape=[None, 1])
        # 词向量矩阵，初始时为均匀随机正态分布
        self.embedding_dict = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
        # 模型内部参数矩阵，初始为截断正太分布
        nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

        # 将输入序列向量化
        embed = tf.nn.embedding_lookup(self.embedding_dict, self.place_x)  # batch_size

        # 得到NCE损失，重要概念NCE损失
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weight,  # 权重
                biases=nce_biases,  # 偏差
                labels=self.place_y,  # 输入的标签
                inputs=embed,  # 输入向量
                num_sampled=self.num_sampled,  # 负采样的个数
                num_classes=self.vocab_size  # 类别数目
            )
        )

        # 根据 nce loss 来更新梯度和embedding，使用梯度下降法(gradient descent)来实现
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)  # 训练操作

        # 计算与指定若干单词的相似度
        self.test_word_id = tf.placeholder(tf.int32, shape=[None])
        vec_l2_model = tf.sqrt(  # 求各词向量的L2模
            tf.reduce_sum(tf.square(self.embedding_dict), axis=1, keep_dims=True)
        )

        self.normed_embedding = self.embedding_dict / vec_l2_model
        # self.embedding_dict = norm_vec # 对embedding向量正则化
        test_embed = tf.nn.embedding_lookup(self.normed_embedding, self.test_word_id)
        self.similarity = tf.matmul(test_embed, self.normed_embedding, transpose_b=True)  # 计算词和次之间的余弦距离，原始的embedding是没有normed的

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train_by_sentence(self, input_sentence):
        # input_sentence: [sub_sent1, sub_sent2, ...]
        # 每个sub_sent是一个单词序列，例如['这次','大选','让']
        batch_inputs = []
        batch_labels = []
        for sent in input_sentence:
            for i in range(sent.__len__()):
                start = max(0, i - self.window_size)
                end = min(sent.__len__(), i + self.window_size + 1)
                for index in range(start, end):
                    if index == i:
                        continue
                    else:
                        input_id = self.word2id.get(sent[i])
                        label_id = self.word2id.get(sent[index])
                        if not (input_id and label_id):
                            continue
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
        if len(batch_inputs) == 0:
            return
        batch_inputs = np.array(batch_inputs, dtype=np.int32)
        batch_labels = np.array(batch_labels, dtype=np.int32)
        batch_labels = np.reshape(batch_labels, [batch_labels.__len__(), 1])

        feed_dict = {
            self.place_x: batch_inputs,
            self.place_y: batch_labels
        }
        _, loss_val = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss_val

    def cal_similarity(self, test_word_id_list, top_k=10):
        sim_matrix = self.sess.run(self.similarity, feed_dict={self.test_word_id: test_word_id_list})
        sim_mean = np.mean(sim_matrix)
        sim_var = np.mean(np.square(sim_matrix - sim_mean))
        test_words = []
        near_words = []
        for i in range(test_word_id_list.__len__()):
            test_words.append(self.vocab_list[test_word_id_list[i]])
            nearst_id = (-sim_matrix[i, :]).argsort()[1:top_k + 1]
            nearst_word = [self.vocab_list[x] for x in nearst_id]
            near_words.append(nearst_word)
        return test_words, near_words, sim_mean, sim_var

    def get_embeddings(self):
        res = self.sess.run(self.embedding_dict)
        return res


if __name__ == '__main__':
    s = list(map(lambda i: chr(i + ord('a')), range(26))) * 100  # 一个长长的句子
    window_size = 2  # 窗口大小
    dic = dict()
    for i in s:
        if i not in dic:
            dic[i] = len(dic)
    reverse_dic = dict((v, k) for k, v in dic.items())
    word_count = len(dic)
    word_count = collections.Counter(s)
    word_count = word_count.most_common(len(s))  # 通过most_common可以获取到出现最多的前几个
    print(word_count)
    word_list = [x[0] for x in word_count]
    print("文本中词汇量", len(word_count))
    # 创建模型，训练
    w2v = Word2Vec(vocab_list=word_list,  # 词典集
                   embedding_size=2,
                   window_size=2,
                   learning_rate=1,  # skip-gram的学习率通常需要设置非常大才有效果
                   num_sampled=10)  # tensorboard记录地址

    for i in range(int(1e8)):
        l = w2v.train_by_sentence([s])
        if l < 0.2:
            break
        if i % 100 == 0:
            print('epoch', i, 'loss', l)
            res = w2v.get_embeddings()
        if i % 500 == 0:
            import pylab as plt

            plt.scatter(res[:, 0], res[:, 1])
            for index, (x, y) in enumerate(res):
                plt.text(x, y + 0.05, reverse_dic.get(index))
            plt.show()
    print(w2v.get_embeddings())
