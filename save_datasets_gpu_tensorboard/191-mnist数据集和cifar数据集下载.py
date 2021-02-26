import sys
import os
import tensorflow as tf
import urllib


class Cifar10:
    SOURCE_URL = 'http://www.cs.toronto.edu/~kriz/'
    WORK_DIRECTORY = 'CIFAR10_data'
    FILE_LIST = ('cifar-10-binary.tar.gz',)


class Mnist:
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    WORK_DIRECTORY = 'MNIST_data'
    FILE_LIST = ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')


# 下载状况
def report_hook(count, block_size, total_size):
    percent = 100.0 * count * block_size / total_size
    sys.stdout.write("\r%02d%%" % int(percent <= 100 and percent or 100) + ' complete')
    sys.stdout.flush()


def download_file(dirpath, filename, url):
    # 判断文件路径是否存在
    if not tf.gfile.Exists(dirpath):
        tf.gfile.MakeDirs(dirpath)
    filepath = os.path.join(dirpath, filename)
    # 判断文件是否存在、需要下载
    if not tf.gfile.Exists(filepath):
        print(filename)
        filepath, _ = urllib.request.urlretrieve(url + filename, filepath, reporthook=report_hook)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print(('Successfully downloaded', filename, size, 'bytes.'))
    return filepath


for dataset in [Mnist, Cifar10]:
    for filename in dataset.FILE_LIST:
        download_file(dataset.WORK_DIRECTORY, filename, dataset.SOURCE_URL)
