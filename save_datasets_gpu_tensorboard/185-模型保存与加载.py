import tensorflow as tf

modelpath = "model/baga"


def train():
    # 训练神经网络
    a = tf.Variable(tf.random_normal([1]), name="a")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([a]))
        print(a.name)
        sess.run(tf.assign(a, [100]))  # 直接给a赋值，就当是训练完成了
        saver.save(sess, modelpath)


def test():
    train()
    # 必须得要重置，否则a变量会定义成a:1，因为a变量已经定义过一次了
    tf.reset_default_graph()
    a = tf.Variable(tf.random_normal([1]), name='a')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 这里无需运行初始化操作，即便运行了也会被覆盖
        saver.restore(sess, modelpath)
        print(sess.run(a))


def test_without_define():
    # 无需定义图结构，直接读取图
    train()
    tf.reset_default_graph()
    # meta文件保存的是图结构
    saver = tf.train.import_meta_graph(modelpath + ".meta")
    with tf.Session() as sess:
        saver.restore(sess, modelpath)
        g = tf.get_default_graph()
        print(tf.global_variables())
        print(sess.run(g.get_tensor_by_name("a:0")))
        print(sess.run(g.get_tensor_by_name("a")))


def test_rename():
    # 在写出、读入时可以对变量进行重命名
    a = tf.Variable(tf.random_normal([1]))  # 默认名称为Variable
    saver = tf.train.Saver({"a": a})  # 将变量a重命名为a
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(a, [100]))  # 直接给a赋值，就当是训练完成了
        saver.save(sess, modelpath)

    tf.reset_default_graph()

    b = tf.Variable([0], dtype=tf.float32)  # 要注意类型一致，否则加载进来读取也是错误的
    saver = tf.train.Saver({"a": b})  # 将变量a加载到b
    with tf.Session() as sess:
        saver.restore(sess, modelpath)
        print(sess.run([b]))


# test()
test_without_define()
# test_rename()
