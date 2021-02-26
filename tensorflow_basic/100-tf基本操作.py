import numpy as np
import tensorflow as tf


def constant():
    a = tf.constant(2)
    b = tf.constant(3)

    with tf.Session() as sess:
        print("a=2, b=3")
        print("Addition with constants: %i" % sess.run(a + b))
        print("Multiplication with constants: %i" % sess.run(a * b))


def placeholder_and_op():
    a = tf.placeholder(tf.int16)
    b = tf.placeholder(tf.int16)

    add = tf.add(a, b)
    mul = tf.multiply(a, b)

    with tf.Session() as sess:
        print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
        print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))


def matmul():
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)

    with tf.Session() as sess:
        result = sess.run(product)
        print(result)


def multiply():
    input1 = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)
    add = tf.add(input2, input3)
    mul = tf.multiply(input1, add)

    with tf.Session() as sess:
        result = sess.run([mul, add])
        print(result)


def use_place_holder():
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, (2, 2), "x")
    y = tf.placeholder(tf.float32, (2, 2), "y")
    why = tf.multiply(x, y)
    with tf.Session() as sess:
        one = np.random.random((2, 2))
        two = np.random.random((2, 2))
        print(one, two)

        print(sess.run([why], feed_dict={
            "x:0": one,
            "y:0": two,
        }))