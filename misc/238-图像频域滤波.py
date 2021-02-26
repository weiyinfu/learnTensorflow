"""
利用卷积可以实现图像的模糊/锐化/提取边缘等

"""
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.data import astronaut
from skimage.color import rgb2gray
import numpy as np
import math

original_img = rgb2gray(astronaut())
img = tf.constant(original_img)
fft_img = tf.fft2d(original_img)

fft_img_len = tf.abs(fft_img)
fft_img_frequency = tf.angle(fft_img)

ifft_img = tf.ifft2d(fft_img)
ifft_img_len = tf.abs(ifft_img)


def filt(mask):
    return tf.ifft2d(tf.where(mask, tf.ones_like(fft_img_len, dtype=tf.complex64), tf.zeros_like(fft_img_len, dtype=tf.complex64)) * fft_img)


"""
既然说到过滤,那肯定是去除少部分而保留大多数
"""
# 过滤掉低能信号:相当于提取边缘/锐化
remove_low_energy = filt(fft_img_len > 1.01 * tf.reduce_min(fft_img_len))
# 过滤掉高能信号:相当于去掉边缘/钝化/模糊
remove_high_energy = filt(fft_img_len < 0.9 * tf.reduce_max(fft_img_len))
remove_low_frequency = filt(fft_img_frequency > 1.01 * tf.reduce_min(fft_img_frequency))
remove_high_frequency = filt(fft_img_frequency < 0.9 * tf.reduce_max(fft_img_frequency))

imgs = [img, fft_img, ifft_img_len, remove_low_energy, remove_high_energy, remove_low_frequency, remove_high_frequency]
titles = ["original", 'fft', 'ifft', 'remove low energy', 'remove high energy', 'remove high frequency', 'remove low frequency']

imgs = [tf.squeeze(i) for i in imgs]
for i in range(len(imgs)):
    if imgs[i].dtype == tf.complex64:
        imgs[i] = -tf.log(tf.abs(imgs[i]))
        print(titles[i], 'is complex')

with tf.Session() as sess:
    a = sess.run(imgs)
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
