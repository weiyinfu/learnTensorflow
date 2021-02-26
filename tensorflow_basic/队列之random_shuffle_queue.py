import tensorflow as tf

"""
把一个序列放进随机队列，再拿出来，这些数据就变成了无序了
"""
it = tf.data.Dataset.range(10).map(lambda x: tf.cast(x, tf.int32)).make_one_shot_iterator()
q = tf.RandomShuffleQueue(10, min_after_dequeue=0, dtypes=[tf.int32], shapes=[tuple()])
it_next = it.get_next()
print(it_next)
enq = q.enqueue(it_next)
qsize = q.size()
deq = q.dequeue()
with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    for i in range(10):
        print(sess.run([enq, qsize]))
    while sess.run(qsize):
        print(sess.run(deq))
