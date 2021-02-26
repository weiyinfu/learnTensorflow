import math

import numpy as np
import tensorflow as tf

"""
conv2d接受4个参数：
* input是一个四维的张量，第一维表示样本，第二维表示宽度，第三维表示高度，第四维表示深度
* w是一个四维的张量，第一维表示卷积核的宽度，第二维表示卷积核的高度，第三维是固定的，等于输入层的深度，tensorflow不支持在深度上进行卷积。第四维表示输出深度。
* strides是一个四元组，表示每个轴上卷积核的移动步长。第一维肯定是1，第二维和第三维表示移动步长，第四维也必然是1。因为在深度方向上是全连接的，没有必要进行移动窗口。

pool层不改变深度，只改变长宽
"""
n = 1  # 样本个数
width = 4  # 图片宽度
height = 4  # 图片高度
d = 2  # 图片深度
a = np.random.randint(0, 4, (n, width, height, d))

"""
strides第一维必须是1，否则没有意义，表示跨样本学习

tensorflow的stride很奇葩
在conv1d中，strides是一个int值，表示只在一个方向上执行strides
可是在conv2d里面，却平白变成了4个值，并且第一维和第四维都必须是写死的1，所以实际上只有第2、第3两维起作用
"""


def conv2d(matrix, filter, strides, padding):
    # 这个函数好复杂呀，更别说用tensorflow实现了
    if padding not in ('SAME', 'VALID'):
        raise Exception("unkown padding {}".format(padding))
    if padding == 'SAME':
        """
        如果是SAME需要在四面补0
        先计算一共有多少个stride，math.ceil(matrix.shape[1] / strides[1]) 
        这些个stride里面最后一个stride需要放满
        最后一个stride为：filter.shape[1]
        前面的stride所占空间为：(math.ceil(matrix.shape[1] / strides[1]) - 1) * strides[1]
        最终填充之后的空间为：pad_width = (math.ceil(matrix.shape[1] / strides[1]) - 1) * strides[1] + filter.shape[0]
        
        有时，当filter较小而stride较大时，求出来的pad_width小于原尺寸，这是不对的，需要矫正一下：pad_width = max(pad_width, matrix.shape[1])
        需要让填充之后的空间居中对齐原空间
        所以需要在左边填充 (pad_width - matrix.shape[1]) // 2
        在右边填充pad_width - x_pad - matrix.shape[1]
        """
        final_width = (math.ceil(matrix.shape[1] / strides[1]) - 1) * strides[1] + filter.shape[0]
        final_height = (math.ceil(matrix.shape[2] / strides[2]) - 1) * strides[2] + filter.shape[1]
        pad_width = max(final_width, matrix.shape[1])
        pad_height = max(final_height, matrix.shape[2])
        x_pad = (pad_width - matrix.shape[1]) // 2
        y_pad = (pad_height - matrix.shape[2]) // 2
        x_pad2 = pad_width - x_pad - matrix.shape[1]
        y_pad2 = pad_height - y_pad - matrix.shape[2]
        matrix = np.pad(matrix, ((0, 0), (x_pad, x_pad2), (y_pad, y_pad2), (0, 0)), mode='constant', constant_values=0)
    width = math.ceil((matrix.shape[1] - filter.shape[0] + 1) / strides[1])
    height = math.ceil((matrix.shape[2] - filter.shape[1] + 1) / strides[2])
    a = np.zeros((matrix.shape[0], width, height, filter.shape[3]), dtype=np.float32)
    for sample in range(a.shape[0]):
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                for deep in range(a.shape[3]):
                    x_beg = i * strides[1]
                    y_beg = j * strides[2]
                    x_end = x_beg + filter.shape[0]
                    y_end = y_beg + filter.shape[1]
                    sub = matrix[sample, x_beg:x_end, y_beg:y_end, :]
                    a[sample, i, j, deep] = np.vdot(sub, filter[:, :, :, deep])
    return a


def get_ans(input, filter, strides, padding):
    tf.reset_default_graph()
    x = tf.constant(input, dtype=tf.float32)
    w = tf.Variable(filter, dtype=tf.float32)  # w卷积第二维和第三维必须相同
    y = tf.nn.conv2d(input=x, filter=w, strides=list(strides), padding=padding)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(y)


for width_stride in range(1, width):
    for height_stride in range(1, height):
        for filter_width in range(1, width):
            for filter_height in range(1, height):
                for filter_depth in range(1, 3):
                    for padding in "SAME VALID".split():
                        filter = np.random.randint(0, 3, (filter_width, filter_height, d, filter_depth))
                        strides = [1, width_stride, height_stride, 1]
                        ans = get_ans(a, filter, strides, padding)
                        mine = conv2d(a, filter, strides, padding)
                        assert ans.shape == mine.shape
                        assert np.all(ans == mine)
