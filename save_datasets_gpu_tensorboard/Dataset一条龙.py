"""
通过dataset可以直接给variable赋值，相当于feedback
这种写法并不提倡，但是这种写法很好玩。直接限制global_step,不管epoch

这种写法对于epoch的干预有点少
"""
import tensorflow as tf

sample_count = 150
feature_count = 3
target_count = 4

batch_size = 32

it = tf.data.Dataset.from_tensor_slices({"x": tf.random_uniform((sample_count, feature_count)), "y": tf.cast(tf.random_uniform((sample_count, 1)) * target_count, tf.int32)}).shuffle(32).repeat().batch(batch_size).make_initializable_iterator()

it_next = it.get_next()
print(it_next, type(it_next))
x_place = it_next['x']
y_place = it_next['y']
logits = tf.layers.dense(x_place, 50, activation='sigmoid')
logits = tf.layers.dense(logits, target_count)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y_place, (-1,)), logits=logits))
y_mine = tf.argmax(logits, axis=1, output_type=tf.int32)
is_right = tf.equal(y_mine, tf.reshape(y_place, (-1,)))
accuracy = tf.reduce_mean(tf.cast(is_right, dtype=tf.float32))
global_step = tf.Variable(0, trainable=False)
train_op = tf.train.AdamOptimizer(0.1).minimize(loss, global_step)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(it.initializer)
    patience = 10
    good = 0
    while good < patience:
        _, lo, acc, step = sess.run((train_op, loss, accuracy, global_step))
        print('step', step, 'loss', lo, 'accuracy', acc)
        if acc > 0.90:
            good += 1
        else:
            good = 0
