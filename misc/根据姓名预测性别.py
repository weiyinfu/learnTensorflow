import collections

import numpy as np
import tensorflow as tf

"""
根据姓名预测性别
这个程序只是一个玩具，大部分情况下人们是难以通过姓名来判断性别的。
网络结构：词嵌入将姓名转化为向量

* 似乎很容易过拟合，训练一轮准确度比较好，训练轮数一多就不行了
"""

###########第一部分：加载数据#############
train_x = []
train_y = []
names="""
姓名,性别
周笑冉,女
毛丹璎,女
郭展成,男
苑永锋,男
向启奖,男
孙睿轩,男
蔡颛岭,男
孟庆安,男
殷俊彪,男
傅木兴,男
孔德然,男
邱静岚,男
房付会,男
金乐瑶,男
黎珀豫,男
吴佩懿,女
许艳旬,女
邬武军,男
付学旭,女
邱文章,男
毛继开,男
傅咝敏,男
毕傲洋,男
裴午顾,男
安镶怡,女
饶黎明,男
段焙曦,男
苗芯萌,男
覃慧藐,女
芦玥微,女
苏佳琬,女
王旎溪,女
彭琛朗,男
李昊,男
利欣怡,女
施杨华,男
董彩富,男
严辽鉴,男
匡佳捷,男
段一轩,男
尹立恩,男
范艺沣,女
徐勇斌,男
喻湘祺,男
唐海柳,女
吕妮甜,女
庞杭军,男
殷璎亭,女
伏丽菲,女
季钰凡,女
荀熙雯,女
阎碌硖,女
巩薪蓣,男
蒲霞箐,女
付灵捷,男
""".split()
for line in names:
    sample = line.strip().split(',')
    train_x.append(sample[0])
    if sample[1] == '男':
        train_y.append([0, 1])  # 男
    elif sample[1] == "女":
        train_y.append([1, 0])  # 女
    else:
        raise Exception(line + " what's the fuck")
print("加载数据完毕", len(train_x))
# 需要求出最长的名字，用来对齐，因为训练时输入的长度是固定的
max_name_length = max(len(name) for name in train_x)
print("最长名字的字符数: ", max_name_length)
max_name_length = 8  # 此处是何意?

# 词汇表，统计名字中出现的全部汉字及其出现次数
vocabulary_counter = collections.Counter("".join(train_x))
vocabulary = sorted(vocabulary_counter.keys(),
                    key=vocabulary_counter.get,
                    reverse=True)

max_vocabulary = 4000  # 最大词汇量：如果超出此词汇量，则选取其中出现频率较多的词汇
vocabulary = vocabulary[:max_vocabulary]
print('词汇表中词汇数量', len(vocabulary))
print("名字中最常出现的前10个汉字", vocabulary[:10])

# 字符串转为向量形式，将全部名字变成等长的向量
# 需要注意，空格为0，所以需要把0留出来
vocab = dict(zip(vocabulary, range(1, 1 + len(vocabulary))))


def transform(x):
    xx = np.zeros((len(x), max_name_length), dtype=np.float32)
    for i, name in enumerate(x):
        for j, c in enumerate(name):
            xx[i][j] = vocab.get(c, 0)  # 不存在的字符记为0
    return xx


train_x = transform(train_x)
train_y = np.array(train_y)
# 对数据打乱顺序
shuffle_indices = np.random.permutation(np.arange(len(train_y)))
train_x = train_x[shuffle_indices]
train_y = train_y[shuffle_indices]
######################神经网络部分#################################
input_size = max_name_length
num_classes = 2

epoch_count = 1
batch_size = 64
model_path = "../model/name2sex.model"

X = tf.placeholder(tf.int32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, num_classes])
dropout_keep_prob = tf.placeholder(tf.float32)

vocabulary_size = len(vocabulary) + 1
embedding_size = 128  # 词嵌入之后词空间的维度
num_filters = 128  # 卷积过滤器的个数，也就是深度
# embedding layer，词嵌入模型，将词汇变为embedding_size维的向量
# 词嵌入模型认为词空间是一个巨大的高维空间
with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embedded_chars = tf.nn.embedding_lookup(W, X)
    # 展成一维
    embedded_chars_expanded = tf.expand_dims(embedded_chars, axis=-1)
# convolution + maxpool layer，三个卷积神经网络并板
filter_sizes = [3, 4, 5]
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        filter_shape = [filter_size,  # 卷积核的大小，也就是对几个字符进行卷积
                        embedding_size,  # 图片长度，也就是每个字符的长度
                        1,  # 卷积输入的深度
                        num_filters  # 卷积深度，三层卷及神经网络各层深度相等
                        ]
        # 卷积矩阵
        W = tf.Variable(tf.truncated_normal(filter_shape,
                                            stddev=0.1))
        b = tf.Variable(tf.constant(0.1,
                                    shape=[num_filters]))
        conv = tf.nn.conv2d(embedded_chars_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID")
        h = tf.nn.relu(tf.nn.bias_add(conv, b))
        pooled = tf.nn.max_pool(h,
                                # ksize的input_size-filter_size+1表示对一个字符进行卷积
                                ksize=[1, input_size - filter_size + 1, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID')
        pooled_outputs.append(pooled)

num_filters_total = num_filters * len(filter_sizes)
# h_pool = tf.concat(3, pooled_outputs)
h_pool = tf.concat(pooled_outputs, axis=3)  # 将三个CNN展成一维
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
# dropout
with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
# output，最后加上一层全连接
with tf.name_scope("output"):
    W = tf.get_variable("W", shape=[num_filters_total, num_classes],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    output = tf.nn.xw_plus_b(h_drop, W, b)  # 一句话相当于wx+b


# 训练
def train_neural_network():
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,
                                                                  labels=Y))
    accuracy = tf.reduce_sum(tf.cast(
        tf.equal(tf.argmax(output, axis=1), tf.argmax(Y, axis=1)), tf.float32))

    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epoch_count):
            for i in range(0, len(train_x), batch_size):
                batch_x = train_x[i:i + batch_size]
                batch_y = train_y[i:i + batch_size]
                sess.run(train_op, feed_dict={X: batch_x,
                                              Y: batch_y,
                                              dropout_keep_prob: 0.5})
            if e % 3 == 0:
                s = 0
                for i in range(0, len(train_x), batch_size):
                    batch_x = train_x[i:i + batch_size]
                    batch_y = train_y[i:i + batch_size]
                    acc = sess.run(accuracy,
                                   feed_dict={
                                       X: batch_x,
                                       Y: batch_y,
                                       dropout_keep_prob: 1
                                   })
                    s += acc
                s /= len(train_x)
                print("周期", e, '精确度', s)
                if s > 0.9:
                    break
        saver.save(sess, model_path)


# 使用训练的模型
def detect_sex(name_list):
    x = transform(name_list)
    # 即便是在预测时，依旧可以定义张量
    predictions = tf.argmax(output, axis=1)
    # 可以只保存一部分张量，因为train_op在测试时根本用不上
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 恢复前一次训练
        saver.restore(sess, model_path)
        res = sess.run(predictions, {X: x, dropout_keep_prob: 1.0})
        for i, name in enumerate(name_list):
            print(name, '女男'[res[i]])


# train_neural_network()
detect_sex(["白富美", "高帅富", "王婷婷",
            "田野", "魏印福", "苏君君", '陈驰',
            "李鹏", "彭丽媛", "段誉", "木婉清",
            "钟万仇", "杜诗宜", '钟灵',
            '李云龙', '贾宝玉', '林黛玉'])
