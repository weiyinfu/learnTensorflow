"""
tensorflow默认运行main函数，tf.app.run会自动运行名称为main的函数，
也可以指定函数名
"""
import tensorflow as tf


def f(x):
    # f函数必须接受一个参数
    print(x, type(x))


if __name__ == "__main__":
    tf.app.run(f)
