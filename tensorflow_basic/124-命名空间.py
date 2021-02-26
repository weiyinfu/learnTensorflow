import tensorflow as tf

"""
tensorflow是一个图模型
是一种编程语言，图中有许多操作结点组成，操作结点的操作对象是张量。
每个操作结点都有一个name属性，每个结点都有名字。
像C++一样有namespace的概念

"""


def case0():
    """
    对于tf.get_variable()形式创建的变量，tf.name_scope()不会给变量名加前缀，
    tf.variable_scope()会给变量名加前缀。

    对于其他的结点的创建形式，tf.name_scope()和tf.variable_scope都会给变量加前缀
    """
    with tf.name_scope("hello") as name_scope:
        arr1 = tf.get_variable("arr1", shape=[2, 10], dtype=tf.float32)
        var2 = tf.Variable(0)
        print(name_scope)  # hello/
        print(arr1.name)  # arr:0
        print(var2.name)  # hello/Variable:0
        print("scope_name:%s " % tf.get_variable_scope().original_name_scope)


def case1():
    with tf.variable_scope("hello") as variable_scope:
        arr1 = tf.get_variable("arr1", shape=[2, 10], dtype=tf.float32)

    print(variable_scope)
    print(variable_scope.name)  # 打印出变量空间名字
    print(arr1.name)  # hello/arr1:0
    print(tf.get_variable_scope().original_name_scope)
    # tf.get_variable_scope() 获取的就是variable_scope

    with tf.variable_scope("xixi"):
        print(tf.get_variable_scope().original_name_scope)
        # tf.get_variable_scope() 获取的就是v _scope2


def case3():
    with tf.name_scope("name1"):
        with tf.variable_scope("var1"):
            w = tf.get_variable("w", shape=[2])
            res = tf.add(w, [3])

    print(w.name)  # var1/w:0
    print(res.name)  # name1/var1/Add:0


def case4():
    def inner():
        # 此处reuse不管是True还是False，最终结果都是新创建变量
        # 这是由tf.Variable()决定的，要想重用，必须使用tf.get_variable()
        with tf.variable_scope("one", reuse=True) as scope:
            x = tf.Variable(0, name='x')
            print(x.name)
            return x

    one = inner()
    two = inner()
    three = inner()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(one, 3))
        print(sess.run([one, two, three]))


def case5():
    """
    对于tf.variable_scope(scopeName,reuse)
    如果把reuse设置为0，则报错ValueError: The reuse parameter must be True or False or None.

    而实际上，对于下面的代码，reuse=False，会导致变量重定义错误；
    reuse=True，会报错，无法使用已定义的变量x
    只有reuse=tf.AUTO_REUSE才能够在没有时新建，有时重用。
    :return:
    """
    print(type(tf.AUTO_REUSE))

    def inner():
        with tf.variable_scope("one", reuse=tf.AUTO_REUSE):
            ini = tf.constant(3.0)
            x = tf.get_variable("x", dtype=tf.float32, initializer=ini)
            print(x.name)
            return x

    one = inner()
    two = inner()
    three = inner()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(one, 3))
        print(sess.run([one, two, three]))


def case6():
    a = tf.constant(3)  # name=Const:0
    b = tf.Variable(4)  # name=Variable:0
    print(a.name, b.name)


def case7():
    with tf.name_scope("my"):
        a = tf.constant(3)  # my/Const:0
        b = tf.add(a, b)  # my/Add:0
        print(a.name, b.name)
        # 使用get_variable却不管用
        c = tf.get_variable("c", shape=1, dtype=tf.int32, initializer=tf.constant_initializer(2))  # c:0
        print(c.name)


def case8():
    with tf.variable_scope("ha"):
        a = tf.constant(2, name="myconstant")  # ha/myconstant:0
        b = tf.get_variable("b", shape=1, dtype=tf.int32, initializer=tf.constant_initializer(2))  # ha/b:0
        print(a.name, b.name)

    with tf.variable_scope("ha", default_name="what", reuse=True):
        try:
            m = tf.get_variable("m")
        except Exception as ex:
            print(ex)  # Variable ha/m does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?


def case9():
    """
    使用tf.variable_scope时，reuse=True、False即将废弃
    改为鼓励使用int值来表示：
    * AUTO_REUSE=1
    * REUSE_FALSE = 2
    * REUSE_TRUE = 3

    这可真是脑残的设计
    """
    with tf.variable_scope("ha", default_name="what", reuse=tf.AUTO_REUSE):
        try:
            m = tf.get_variable("m")
            print(m.name)
        except Exception as ex:
            print(ex)  # ValueError: Shape of a new variable (ha/m) must be fully defined, but instead was <unknown>.  这个错误是在说：创建变量必须指明变量的类型
        # 当variable_scope reuse变量时，依旧可以对变量进行一些微操作：设置trainable=False，表示这个节点不可训练
        # 当获取变量时，dtype类型必须对应正确
        a = tf.get_variable("b", trainable=False, dtype=tf.int32)
        print(tf.trainable_variables("ha"))


def case10():
    # 变量作用域嵌套
    with tf.variable_scope("one"):
        """
        使用name_scope只会影响变量的名字，它要解决的问题是变量重名问题
        """
        with tf.name_scope("two"):
            x = tf.constant(3)  # one/two/Const:0
            print(x.name)
            # variable_scope.name是根目录的名字
            print(tf.get_variable_scope().name, tf.get_variable_scope().original_name_scope)  # 输出为：one    one/
        with tf.variable_scope("three"):
            x = tf.constant(3)  # one/three/Const:0
            print(x.name)
            print(tf.get_variable_scope().name, tf.get_variable_scope().original_name_scope)  # 输出为one/three  one/three/


def case11():
    # 使用函数依旧不会影响作用域
    def ha():
        x = tf.Variable(3)  # ha_3/Variable:0
        print(x.name)

    with tf.variable_scope("ha"):  # 这个变量作用域已经定义过好几次了，它的实际名字变成了ha_3
        ha()


def case12():
    # 如果重复定义变量
    a = tf.constant(2, name='a')  # a:0
    b = tf.constant(2, name='a')  # a_1:0
    print(a.name, b.name)

    # 如果重复定义name_scope
    with tf.name_scope("my"):
        a = tf.constant(2)  # my_1/Const:0
        print(a.name)
    """
    可见tensorflow对于一切重名的东西都会在末尾加上下划线+数字
    """
    with tf.name_scope("my_4"):
        a = tf.constant(2)
        print(a.name)  # my_4/Const:0
    with tf.name_scope("my"):
        a = tf.constant(2)
        print(a.name)  # my_2/Const:0
    with tf.name_scope("my_4"):
        a = tf.constant(2)  # my_4_1/Const:0
        print(a.name)


"""
通过以上例子可以发现，tensorflow对于命名重复问题使用以下规则解决：
1、要使用的name不存在，可以直接使用
2、要使用的name已经存在，执行下列循环：
i=1
while 1：
    now_name="%s_%d"%(name,i)
    if exists(now_name):
       i+=1
    else:
        return now_name
"""


def case13():
    # 下面我们来验证一下变量共享机制
    def get_share_variable(reuse):
        with tf.variable_scope("share", reuse=reuse):
            a = tf.get_variable("a", shape=(1), dtype=tf.int32)
            return a

    one = get_share_variable(False)
    two = get_share_variable(True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([one, two, tf.assign(one, [2])]))


def case14():
    # reuse变量的唯一方式就是使用name_scope
    x = tf.Variable(3, False, name='x')
    print(x.name, x.shape)  # x:0 ()
    y = tf.get_variable("x", shape=x.shape)
    print(y.name, y.shape)  # x_1:0 ()
