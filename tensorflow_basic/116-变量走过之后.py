"""
使用sess.run([x,y,z])时，实际上只计算一次，不过是返回值多了几个罢了
"""
import tensorflow as tf

y = tf.Variable(1, name="y")
x = tf.add(y, y, name="y2")
x = tf.add(x, x, name='y4')
x = tf.add(x, x, name='y8')
z = tf.group(x, tf.assign(y, x))
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(4):
    print(sess.run([x, y, z]))
sess.close()
