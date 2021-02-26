import tensorflow as tf
import numpy as np

sess = tf.Session()
x = np.arange(20)
input_ta = tf.TensorArray(size=0, dtype=tf.int32, dynamic_size=True)
input_ta = input_ta.unstack(x)  # TensorArray可以传入array或者tensor
for i in range(len(x)):
    print(sess.run(input_ta.read(i)))  # 遍历查看元素

output = input_ta.stack()  # 合成
print(sess.run(output))

for i in range(5):
    input_ta = input_ta.write(i + len(x), i)  # 写入

output = input_ta.stack()
print(sess.run(output))

a = tf.TensorArray(tf.float32, 0, True)
for i in range(9):
    a = a.write(i, i * 1.0)
a = a.stack()
a = tf.reshape(a, (3, 3))
with tf.Session() as sess:
    print(sess.run(a))

"""
关于tensorArray只需要知道以下四种操作:
ta.stack(name=None) 将TensorArray中元素叠起来当做一个Tensor输出
ta.unstack(value, name=None) 可以看做是stack的反操作，输入Tensor，输出一个新的TensorArray对象
ta.write(index, value, name=None) 指定index位置写入Tensor
ta.read(index, name=None) 读取指定index位置的Tensor
"""
