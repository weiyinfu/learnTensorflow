from collections import Counter

import tensorflow as tf

from keras.layers import *

"""
奇数为仄，偶数为平，让神经网络学会说 
平平仄仄平
平平平仄仄
仄仄仄平平
仄仄平平仄

"""
sentence_length = 5
word_count = 10
embedding_size = 13

place_idea = Input((sentence_length, embedding_size), name='place_idea')
place_pingze = Input((sentence_length,), dtype=tf.int32, name='place_pingze')
place_word = Input((sentence_length,), dtype=tf.int32, name='place_word')
place_flag = tf.placeholder(dtype=tf.float32, shape=(None,), name='place_flag')

encoded = LSTM(5, return_sequences=True)(place_idea)
words = TimeDistributed(Dense(word_count))(encoded)
words = tf.reshape(words, (-1, word_count))
softmax_words = tf.nn.softmax(words, axis=1)

pingze_embedding = np.zeros((word_count, 2), dtype=np.float32)
pingze_embedding[::2, :] = [1, 0]
pingze_embedding[1::2, :] = [0, 1]
pingze = tf.Variable(pingze_embedding, trainable=False, dtype=tf.float32)  # 两个类别

logits = tf.matmul(softmax_words, pingze)
flat_pingze = tf.reshape(place_pingze, (-1,))
flat_word = tf.reshape(place_word, (-1,))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=flat_pingze))  # 混乱程度应该尽量大
words_loss = tf.reduce_sum(tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_word, logits=words), (tf.shape(place_flag)[0], -1)) * tf.expand_dims(place_flag, 1)) / tf.reduce_sum(place_flag)
loss += words_loss
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
output_pingze = tf.argmax(logits, axis=1, output_type=tf.int32)

accuracy_pingze = tf.reduce_mean(tf.cast(tf.equal(output_pingze, flat_pingze), tf.float32))
output_words = tf.argmax(words, axis=1, output_type=tf.int32)
accuracy_word = tf.reduce_sum(tf.reshape(tf.cast(tf.equal(output_words, flat_word), tf.float32), (tf.shape(place_flag)[0], -1)) * tf.expand_dims(place_flag, axis=1)) / (tf.reduce_sum(place_flag) * sentence_length)


def train(sess, x, y, pingze, flag):
    # 让神经网络学会格式
    patience = 10
    index = np.arange(len(x))
    print(place_flag)
    for epoch in range(1000):
        np.random.shuffle(index)
        _, lo, acc_pingze, acc_word = sess.run([train_op, loss, accuracy_pingze, accuracy_word], feed_dict={
            place_idea: x[index],
            place_pingze: pingze[index],
            place_word: y[index],
            place_flag: flag[index],
        })
        if acc_word == 1.0 and acc_pingze == 1.0:
            if not patience:
                break
            else:
                patience -= 1
        print('epoch', epoch, 'loss', lo, 'accuracy_word', acc_word, 'accuracy_pingze', acc_pingze)


available = np.array([[0, 0, 1, 1, 0], [0, 0, 0, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 1]])


def getdata():
    # 伪造一些数据
    datasize = 10
    x = np.random.random((datasize, sentence_length, embedding_size))
    y = []
    flag = np.random.randint(0, 2, datasize).astype(np.float32)
    pingze = []
    for i in x:
        now_y = available[int(np.mean(i) // 0.25)]
        y.append(np.array(np.random.randint(0, word_count // 2, len(now_y))) * 2 + now_y)
        pingze.append(now_y)
    return np.array(x), np.array(y), np.array(pingze, dtype=np.int32), np.array(flag)


with tf.Session()as sess:
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    x, y, pingze, flag = getdata()
    print(flag.dtype, flag.shape)
    train(sess, x, y, pingze, flag, )
    a = []
    batch_size = 10
    for i in range(10):
        w = sess.run(output_words, feed_dict={
            place_idea: np.random.random((batch_size, sentence_length, embedding_size)),
        })
        w = np.reshape(w, (batch_size, -1))
        for i in w:
            a.append(''.join(str(j) for j in i))
    a = Counter(a)
    for s, cnt in a.items():
        print(s, cnt)
