"""
使用feed_dict，也就是输入参数
终结点除了可以是constant，variable，还可以使用placeholder运行时提供
不指定维度表示任意维度

feed_dict可以作用于：
* variable
* place_holder
* op后的张量
"""
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.Variable(2.0, dtype=tf.float32, validate_shape=False)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    # place_holder可以给variable赋值
    print(sess.run([output], feed_dict={input1: 7, input2: 2}))
    print(sess.run([output], feed_dict={input1: [7], input2: [2]}))
    print(sess.run([output], feed_dict={input1: [7, 2], input2: [2, 1]}))
    # feed_dict直接给op后的结果赋值
    print(sess.run([output], feed_dict={output: 3}))
