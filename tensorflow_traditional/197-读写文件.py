import tensorflow as tf

"""
tensorflow的计算图模型相当于一种新的语言

连文件读取操作都有
"""
filename = "model/place_holder_and_op.txt"
a = tf.Variable(3)
b = tf.Variable(4)
c = tf.Variable(filename, dtype=tf.string)
# write_file和read_file都是一种动作
write_file = tf.write_file(c, tf.as_string(tf.add(a, b)), "write_file")
read_file = tf.string_to_number(tf.read_file(c))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(write_file)  # 返回None
    res = sess.run(write_file)  # 返回None
    print(res)
    print(sess.run([read_file]))  # 7
