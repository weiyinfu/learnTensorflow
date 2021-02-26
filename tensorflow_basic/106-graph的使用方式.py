import tensorflow as tf

a = tf.constant(0)
print(a.graph, tf.get_default_graph())

graph2 = tf.Graph()
with graph2.as_default():
    aa = tf.constant(1)
with tf.Session(graph=tf.get_default_graph()) as sess:
    print(sess.run(a))  # 0
with tf.Session(graph=graph2) as sess:
    print(sess.run(aa))  # 1

print(tf.get_default_graph().get_operations())
tf.reset_default_graph()  # 清空图
print(tf.get_default_graph().get_operations())
