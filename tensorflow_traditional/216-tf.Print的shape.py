import tensorflow as tf

x = tf.Print(['why', "hello", "world"], ['why', "hello", "world"])
print(x.shape)
print(x.dtype)
print(type(x))
with tf.Session() as sess:
    res = sess.run(x)
    for i in res:
        print(str(i, 'utf8'))
