import numpy as np
import tensorflow as tf

"""
tensorflow中的数据类型和numpy是共通的
"""
print(tf.int32 == np.int32)
print(tf.float32 == np.float32)
a = tf.Variable(3)
print(a.dtype, a.dtype == tf.int32, a.dtype is tf.int32)
