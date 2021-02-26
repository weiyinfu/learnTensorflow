import tensorflow as tf

"""
构建一个包含重复元素的数组，然后成批，最后去重
"""
it_next = tf.data.Dataset.range(10).repeat(10).shuffle(32).batch(100).map(lambda x: tf.unique(x)[0]).make_one_shot_iterator().get_next()
with tf.Session()as sess:
    try:
        while 1:
            print(sess.run(it_next))
    except tf.errors.OutOfRangeError:
        pass
