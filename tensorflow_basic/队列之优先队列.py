import tensorflow as tf

"""
打乱一个数组之后输入到优先队列，拿出来的时候就变得有序了。
"""
it_next = tf.data.Dataset.range(10).map(lambda x: tf.cast(x, tf.int32)).shuffle(10).make_one_shot_iterator().get_next()
print(it_next)
q = tf.PriorityQueue(10, types=[tf.int32], shapes=[tuple()])
enq = q.enqueue((tf.cast(it_next, tf.int64), it_next))  # 入队之时必须带着一个优先级tuple，优先级是int64类型
_, deq = q.dequeue()  # 丢弃掉第一个元素，第一个元素表示优先级
qsize = q.size()
with tf.Session()as sess:
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    try:
        while 1:
            sess.run(enq)
    except tf.errors.OutOfRangeError:
        pass
    while sess.run(qsize):
        print(sess.run(deq))
