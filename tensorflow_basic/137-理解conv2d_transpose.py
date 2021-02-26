import tensorflow as tf

x1 = tf.constant(1.0, shape=[1, 3, 3, 1])

x2 = tf.constant(1.0, shape=[1, 6, 6, 3])

x3 = tf.constant(1.0, shape=[1, 5, 5, 3])

# kernel表示:width_kernel,height_kernel,depth_kernel,target_depth_kernel
kernel = tf.constant(1.0, shape=[3, 3, 3, 1])

x1_deconv = tf.nn.conv2d_transpose(x1, kernel, output_shape=[1, 6, 6, 3],
                                   strides=[1, 2, 2, 1], padding="SAME")
print(x1_deconv.shape)# 1,6,6,3
"""
conv2d过程只需要指定原图,kernel和stride,最后生成的图像的尺寸就自动确定了
conv2d_transpose却需要指明output_shape,这是因为反卷积过程即便确定了kernel和stride依旧有多种可能的输出结果,例如
x2和x3卷积过后都能够得到尺寸为1,3,3,1的输出,所以反卷积的时候需要指明output_shape
"""
x3_conv = tf.nn.conv2d(x3, kernel, strides=[1, 2, 2, 1], padding="SAME")
print(x3_conv.shape)
x3_conv_deconv = tf.nn.conv2d_transpose(x3_conv, kernel, output_shape=[1, 5, 5, 3], strides=[1, 2, 2, 1], padding="SAME")
print(x3_conv_deconv.shape)
x2_conv = tf.nn.conv2d(x2, kernel, strides=[1, 2, 2, 1], padding="SAME")
print(x2_conv.shape)
'''
Wrong!!This is impossible
y5 = tf.nn.conv2d_transpose(x1,kernel,output_shape=[1,10,10,3],strides=[1,2,2,1],padding="SAME")
'''
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
x1_decov, x3_cov, y2_decov, x2_cov = sess.run([x1_deconv, x3_conv, x3_conv_deconv, x2_conv])
print(x3_cov)
print(y2_decov)
