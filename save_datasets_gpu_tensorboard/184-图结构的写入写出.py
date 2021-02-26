import tensorflow as tf

"""
tf.train.export_meta_graph
tf.train.import_meta_graph

tf.train.write_graph
tf.import_graph_def

tf.Saver.load()
tf.Saver.save()
"""


def train():
    a = tf.Variable(3, name='a')
    with tf.Session() as sess:
        sess.run(tf.assign(a, 100))
        print(tf.get_default_session() == sess)  # True
        print(tf.get_default_graph() == sess.graph_def)  # False,图和图定义是两码事
        print(tf.get_default_graph() == sess.graph)  # True
        tf.train.export_meta_graph("model/place_holder_and_op.meta", as_text=True)  # 只导出了图结构，没有导出变量值


def use_write_graph():
    """
    tf.train.write_graph默认是以文本方式导出的
    :return:
    """
    a = tf.Variable(3, name='a')
    with tf.Session():
        tf.train.write_graph(tf.get_default_graph(), "model", "write_graph", True)


def test():
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("model/place_holder_and_op.meta")
    print(type(saver))
    with tf.Session() as sess:
        g = tf.get_default_graph()
        print(g.get_operations())
        a = g.get_operation_by_name("a")
        print(a, type(a))


# train()
# test()
use_write_graph()
