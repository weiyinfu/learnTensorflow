import tensorflow as tf
"""
Eager模式的优点包括：

快速debug运行时错误
通过python的控制流，支持动态模型
支持自定义的和高阶的梯度计算
支持大部分TensorFlow的指令

注意事项：
* 你可以为大多数模型写代码，这对 eager execution 和图构建同样有效。也有一些例外，比如动态模型使用 Python 控制流改变基于输入的计算。
* 一旦调用 tfe.enable_eager_execution()，它不可被关掉。为了获得图行为，需要建立一个新的 Python session。

"""
tf.enable_eager_execution()
from tensorflow.contrib.eager.python import tfe as e

x = e.Variable(3.0)


def f(x):
    if x > 2:
        y = x * 3
    else:
        y = x * 2
    return y


ff = e.gradients_function(f)
print(ff(1.0))
print(ff(3.0))

"""
使用注解的方式定义梯度函数
"""


@e.gradients_function
def gg(x, y):
    return x * 4, y * 5


print(gg(1., 1.))
