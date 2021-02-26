import tensorflow as tf

"""
tf.get_variable()和Variable有很多不同点
* 它们对重名操作的处理不同,Variable对于重名操作会默不做声另起别名,tf.get_variable()则会直接抛出异常
* 它们受name_scope的影响不同:tf.get_variable不受name_scope的影响,Variable会受到name_scope的影响
"""
a = tf.Variable(3, name="a")
b = tf.Variable(3, name="a")
print(a.name, b.name)  # a:0 a_1:0

try:
    a = tf.get_variable("b", initializer=tf.constant_initializer(3), shape=1)
    b = tf.get_variable("b", initializer=tf.constant_initializer(3), shape=1)  # Variable b already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
except Exception as ex:
    print(ex)
