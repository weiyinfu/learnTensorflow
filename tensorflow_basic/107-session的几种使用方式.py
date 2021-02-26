"""
tensorflow中session一共有三种用法，但是这些用法根本不需要掌握，只需要知道session.run这一种用法即可
* interactiveSession()，表示默认session
* op.eval(sess)
* sess.run()
"""
import tensorflow as tf

# 第一种方式：使用tf.InteractiveSession()指明默认session
a = tf.constant(3)
b = tf.add(a, a)
sess = tf.InteractiveSession()
print(sess.run(a))
print(b.eval())
sess.close()

# 第二种方式：sess.run()和tensor.eval(session=sess)
tf.reset_default_graph()
n_values = 32
x = tf.linspace(-3.0, 3.0, n_values)
sess = tf.Session()
result = sess.run(x)
x.eval(session=sess)
# x.eval() does not work, as it requires a session!

# %% We can setup an interactive session if we don't
# want to keep passing the session around:
sess.close()
sess = tf.InteractiveSession()

# %% Now this will work!
x.eval()
