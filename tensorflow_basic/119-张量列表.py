import tensorflow as tf

"""
结点有group
边有collection
再定义变量时我们可以指明collection

tf.get_trainable_collections()输出的就是list<>,里面包含的元素有Variable和Tensor


这一点非常重要
张量集合是编译阶段确定的，我们可以在构图的时候随意打印张量集合
"""
x = tf.placeholder(tf.int32)
# 默认会执行所有的add_to_collectioon操作
tf.add_to_collection(name="x", value=x)
tf.add_to_collection(name="x", value=tf.multiply(tf.constant(3), x))
get = tf.get_collection_ref("x")
with tf.Session()as sess:
    for i in range(10):
        sum = sess.run(tf.add_n(get), feed_dict={x: i, })
        print(sum)
        """
        在张量上可以执行各种运算
        """
        l2 = sess.run(tf.sqrt(tf.cast(tf.reduce_mean(tf.pow(get, 2)), dtype=tf.float32)), feed_dict={x: i})
        print(l2)

# 在变量定义时指明集合
a = tf.Variable(0, collections=["x"])
print(tf.get_collection('x'))
for i in tf.get_collection("x"):
    print(i.name, i.shape)

# 张量集合类型
print(tf.trainable_variables(), type(tf.trainable_variables()))
print(tf.get_collection("x"), type(tf.get_collection("x")))
print(tf.global_variables(), type(tf.global_variables()))
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
print(tf.GraphKeys.UPDATE_OPS)
print(dir(tf.GraphKeys))
