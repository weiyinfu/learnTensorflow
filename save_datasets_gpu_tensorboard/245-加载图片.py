import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Hyperparams
batch_size = 10
num_epochs = 1

# Make fake images and save
for i in range(100):
    _x = np.random.randint(0, 256, size=(10, 10, 4))
    plt.imsave("example/image_{}.jpg".format(i), _x)

# Import jpg files
images = tf.train.match_filenames_once('example/*.jpg')

# Create a string queue
fname_q = tf.train.string_input_producer(images, num_epochs=num_epochs, shuffle=True)

# Q10. Create a WholeFileReader
reader = tf.WholeFileReader()

# Read the string queue
_, value = reader.read(fname_q)

# Q11. Decode value
img = tf.image.decode_image(value)

# Batching
img_batch = tf.train.batch([img], shapes=([10, 10, 4]), batch_size=batch_size)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    num_samples = 0
    try:
        while not coord.should_stop():
            sess.run(img_batch)
            num_samples += batch_size
            print(num_samples, "samples have been seen")

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
