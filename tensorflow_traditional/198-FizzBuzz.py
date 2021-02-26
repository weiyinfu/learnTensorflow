"""
TensorFlow 所构成的计算图是图灵完备的。
当我们提到 TensorFlow 的时候，我们仅仅只会关注它是一个很好的神经网络和深度学习的库。但是，它也满足数据流编程（DFP）的各个方面。

由于 TensorFlow 具有 tf.cond 和 tf.while_loop 函数，前者可以处理判断语句，后者可以处理循环语句，所以它就具有一般编程语言相同的表达式。简单的说，我们可以用 C 语言或者 Python 语言实现的排序和搜索算法都可以在 TensorFlow 图中实现。

在本文中，我将介绍的就是 TensorFlow 的另一面，它的一般编程语言表达方式。我利用 TensorFlow 图实现了一些算法，诸如 FizzBuzz，Bubble Sort，Quick Sort，Binary Search 等等。

FizzBuzz是一个小程序，从1开始数数，数到3的倍数的时候输出Fizz，数到5的倍数的时候输出Buzz，如果既是3的倍数又是5的倍数，输出FizzBuzz

"""
import tensorflow as tf


class FizzBuzz():
    def __init__(self, length=30):
        self.length = length  # 程序需要执行的序列长度
        self.array = tf.Variable([str(i) for i in range(1, length + 1)], dtype=tf.string, trainable=False)  # 最后程序返回的结果
        self.my_while = tf.while_loop(self.cond, self.body, [1, self.array], )  # 对每一个值进行循环判断

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.my_while)

    def cond(self, i, _):
        return tf.less(i, self.length + 1)  # 判断是否是最后一个值

    def body(self, i, _):
        flow = tf.cond(tf.equal(tf.mod(i, 15), 0),  # 如果值能被 15 整除，那么就把该位置赋值为 FizzBuzz
            lambda: tf.assign(self.array[i - 1], 'FizzBuzz'),

            lambda: tf.cond(tf.equal(tf.mod(i, 3), 0),  # 如果值能被 3 整除，那么就把该位置赋值为 Fizz
                            lambda: tf.assign(self.array[i - 1], 'Fizz'),
                            lambda: tf.cond(tf.equal(tf.mod(i, 5), 0),  # 如果值能被 5 整除，那么就把该位置赋值为 Buzz
                                            lambda: tf.assign(self.array[i - 1], 'Buzz'),
                                            lambda: self.array  # 最后返回的结果
                                            )
                            )
        )
        return tf.add(i, 1), flow


if __name__ == '__main__':
    fizzbuzz = FizzBuzz(length=50)
    ix, array = fizzbuzz.run()
    print([str(i, "utf8") for i in array])
