"""
利用卷积可以实现图像的模糊/锐化/提取边缘等

"""
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.data import astronaut
from skimage.color import rgb2gray
import numpy as np
import math

"""
laplace提取边缘
"""
laplace = np.array([[1, 1, 1.0], [1, -8.0, 1.0], [1, 1, 1.0]])
# 均值滤波
average = np.ones((3, 3), dtype=np.float32)
# sobel算子边缘提取
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).T

# 原始图片
original_img = rgb2gray(astronaut())
# tensorflow对于图片的卷积必须是rank=4的图片
img = tf.reshape(original_img, (1, *original_img.shape, 1))
# laplace滤波,因为会产生负数,所以需要取模
laplace_image = tf.abs(tf.nn.conv2d(img, np.reshape(laplace, (*laplace.shape, 1, 1)), strides=[1, 1, 1, 1], padding='SAME'))
# 均值滤波
average_image = tf.nn.conv2d(img, np.reshape(average, (*average.shape, 1, 1)), strides=[1, 1, 1, 1], padding='SAME')
# 执行多次次均值滤波
average_image2 = average_image
for i in range(10):
    average_image2 = tf.nn.conv2d(average_image2, np.reshape(average, (*average.shape, 1, 1)), strides=[1, 1, 1, 1], padding='SAME')

# fft
fft_img = tf.fft2d(original_img)
fft_img_len = tf.log(tf.abs(fft_img))
ifft_img = tf.ifft2d(fft_img)
ifft_img_len = tf.abs(ifft_img)

# sobel算子在提取图像边缘方面简直无敌了
sobel_x_img = tf.nn.conv2d(img, np.reshape(sobel_x, (*sobel_x.shape, 1, 1)), strides=[1, 1, 1, 1], padding='SAME')
sobel_y_img = tf.nn.conv2d(img, np.reshape(sobel_y, (*sobel_y.shape, 1, 1)), strides=[1, 1, 1, 1], padding='SAME')
sobel_img = tf.sqrt(sobel_x_img ** 2 + sobel_y_img ** 2)

imgs = [img, laplace_image, average_image, average_image2, fft_img_len, ifft_img_len, sobel_img]
titles = ["original", 'laplace', 'average', 'average 2', 'fft', 'ifft', "sobel"]
# 图片本来是4维,需要把维度降下来
imgs = [tf.squeeze(i) for i in imgs]

with tf.Session() as sess:
    a = sess.run(imgs)
    #计算画图的行和列,尽量让行数和列数相等
    row_count = int(np.sqrt(len(a)))
    col_count = math.ceil(len(a) / row_count)
    fig, demos = plt.subplots(row_count, col_count)
    demos = demos.reshape(-1)
    for i, im in enumerate(a):
        demos[i].imshow(im, cmap='gray')
        demos[i].axis('off')
        demos[i].set_title(titles[i])
    for i in range(len(a), len(demos)):
        demos[i].axis('off')
    plt.show()
