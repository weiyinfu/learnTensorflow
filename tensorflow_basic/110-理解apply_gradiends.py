import tensorflow as tf

"""

class tf.train.Optimizer

操作	描述
class tf.train.Optimizer	基本的优化类，该类不常常被直接调用，而较多使用其子类，
比如GradientDescentOptimizer, AdagradOptimizer
或者MomentumOptimizer
tf.train.Optimizer.__init__(use_locking, name)	创建一个新的优化器，
该优化器必须被其子类(subclasses)的构造函数调用
tf.train.Optimizer.minimize(loss, global_step=None, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None)	添加操作节点，用于最小化loss，并更新var_list
该函数是简单的合并了compute_gradients()与apply_gradients()函数
返回为一个优化更新后的var_list，如果global_step非None，该操作还会为global_step做自增操作

tf.train.Optimizer.compute_gradients(loss,var_list=None, gate_gradients=1,aggregation_method=None, colocate_gradients_with_ops=False, grad_loss=None)	对var_list中的变量计算loss的梯度
该函数为函数minimize()的第一部分，返回一个以元组(gradient, variable)组成的张量列表
tf.train.Optimizer.apply_gradients(grads_and_vars, global_step=None, name=None)	将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作
tf.train.Optimizer.get_name()	获取名称


** AdamOptimizer的apply_gradiends是无效的 **
** GradientDescentOptimizer的apply_gradiends是有效的**

"""
x = tf.Variable(10, dtype=tf.float32)
op = tf.train.GradientDescentOptimizer(learning_rate=1)
app = op.apply_gradients([(tf.Variable(2, dtype=tf.float32), x)])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        _, xx = sess.run([app, x])
        print(xx)
