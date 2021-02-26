import tensorflow as tf

"""
卷积神经网络用于处理图片
一张图片有长度、宽度、深度三个维度
在这三个维度上，本来可以各个维度的kernelsize和stride都随意变动，但一般情况下
长度上的kernelsize和宽度上的kernelsize相等，深度上的kernelsize始终为1
本例仅演示长度、宽度上kernelsize相等、深度上kernelsize为1的情形
stride和kernelsize
"""


def conv(cin, cout, stride=1, kernel_size=3, padding="SAME"):
    """

    :param cin: 输入层
    :param cout: 输出层的深度
    :param stride: 各维移动的长度
    :param kernel_size: 卷积核的大小
    :param padding: 填充方式
    :return: 输出层
    """
    # cin.shape[-1]上一层的深度
    w = tf.Variable(tf.random_normal([kernel_size, kernel_size, int(cin.shape[-1]), cout]))
    b = tf.Variable(tf.random_normal([cout]))
    output = tf.nn.conv2d(input=cin,
                          filter=w,
                          strides=[1, stride, stride, 1],
                          padding=padding)
    output = tf.nn.relu(tf.nn.bias_add(output, b))
    return output


def pool(cin, kernel_size=2, stride=2, padding="SAME"):
    """
    池化层只改变图像的长宽尺寸，不改变图像深度
    :param cin: 输入层神经元
    :param kernel_size: 卷积核的大小
    :param stride: 各维移动的距离
    :param padding: 填充方式
    :return:
    """
    return tf.nn.max_pool(value=cin,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding)


layer_type = "conv pool".split()
stride = [1, 2, 3]
padding = "SAME VALID".split()
kernel_size = [1, 2, 3, 4, 5, 6, 7]

x_place = tf.placeholder(tf.float32, (None, 28, 28, 3))
print("在输入为", x_place.shape, '的情况下')
for layer in layer_type:
    for stri in stride:
        for pad in padding:
            for ksize in kernel_size:
                print("question:", '\n',
                      "layer=", layer, '\n',
                      "stride=", stri, '\n',
                      'padding=', pad, '\n',
                      'kernel_size=', ksize, '\n')
                if layer == 'conv':
                    print('ans:', conv(x_place, cout=2, stride=stri, kernel_size=ksize, padding=pad).shape)
                elif layer == "pool":
                    print('ans:', pool(x_place, stride=stri, padding=pad).shape)
                print("=" * 10)
