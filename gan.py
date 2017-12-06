import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

# Commented out code to visualize the mnist dataset

# sample = mnist.train.next_batch(1)
# sample_image = sample[0]
# sample_label = sample[1]
# print(sample_image.shape, sample_label)
#
# sample_image = sample_image.reshape([28, 28])
# plt.imshow(sample_image, cmap='Greys')
# plt.show()

# Discriminator network

def discriminator(images, reuse=False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()


    # First convolutional and pool layers
    d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
    d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
    d1 = d1 + d_b1
    d1 = tf.nn.relu(d1)
    d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second convolutional and pool layers
    # This fidns 64 different 5 x 5 pixel features
    d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
    d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
    d2 = d2 + d_b2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # First fully connected layer
    d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
    d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
    d3 = tf.matmul(d3, d_w3)
    d3 = d3 + d_b3
    d3 = tf.nn.relu(d3)

    # Second fully connected layer
    d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
    d4 = tf.matmul(d3, d_w4) + d_b4

    # d3 contains unscaled values
    return d4

# Generator network
def generator(z, batch_size, z_dim, reuse=False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()

    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [56, 56])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56, 56])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g4

# Uncomment to view the generator generate images

batch_size = 100
z_dimensions = 100
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

generated_image_output = generator(z_placeholder, 1, z_dimensions)
z_batch = np.random.normal(0, 1, [1, z_dimensions])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    generated_image = sess.run(generated_image_output,
                                feed_dict={z_placeholder: z_batch})
    generated_image = generated_image.reshape([28, 28])
    plt.imshow(generated_image, cmap='Greys')
    plt.show()

tf.reset_default_graph()
batch_size =100
z_dimension = 100

z_placeholder = tf.placeholder(tf.float32,[None,z_dimension],name='z_placeholder')
x_placeholder = tf.placeholder(tf.float32,[None,28,28,1],name='x_placeholder')

with tf.variable_scope(tf.get_variable_scope()):
    Gz = generator(z_placeholder, batch_size, z_dimension)
    # Gz holds the generated images

    Dx = discriminator(x_placeholder)
    # Dx will hold discriminator prediction probabilities
    # for the real MNIST images

    Dg = discriminator(Gz, reuse=True)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(Dx),logits=Dx))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(Dg),logits=Dg))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(Dg),logits=Dg))

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

print ([v.name for v in d_vars])
print ([v.name for v in g_vars])

with tf.variable_scope(tf.get_variable_scope()):
    # Train the discriminator
    d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
    d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

    # Train the generator
    g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

tf.summary.scalar('Discriminator_loss_real',d_loss_real)
tf.summary.scalar('Discriminator_loss_fake',d_loss_fake)
tf.summary.scalar('Generator_loss',g_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

with tf.variable_scope(tf.get_variable_scope()) as scope:
    images_for_tensorboard = generator(z_placeholder, batch_size, z_dimension,reuse=True)
    tf.summary.image('Generated_images', images_for_tensorboard, 5)
    merged = tf.summary.merge_all()
    logdir = 'Tensorboard/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '/'
    writer = tf.summary.FileWriter(logdir,sess.graph)

#Pre-train discriminator
for i in range(3000):
    z_batch = np.random.normal(0, 1, size=[batch_size,z_dimension])
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
    _,__,dLossReal,dLossFake = sess.run([d_trainer_real,d_trainer_fake,d_loss_real,d_loss_fake],
                                       feed_dict={x_placeholder:real_image_batch,z_placeholder:z_batch})
    if(i%100==0):
        print ('dLossReal: ',dLossReal,'dLossFake: ',dLossFake)

#Train discriminator and generator together
for i in range(100000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
    z_batch = np.random.normal(0,1,size=[batch_size,z_dimension])

    _,__,dLossReal,dLossFake = sess.run([d_trainer_real,d_trainer_fake,d_loss_real,d_loss_fake],
                                        feed_dict={x_placeholder:real_image_batch,z_placeholder:z_batch})
    z_batch = np.random.normal(0,1,size=[batch_size,z_dimension])
    _ = sess.run(g_trainer,feed_dict={z_placeholder:z_batch})

    if i%10 ==0:
        z_batch = np.random.normal(0,1,size=[batch_size,z_dimension])
        summary = sess.run(merged,feed_dict={x_placeholder:real_image_batch,z_placeholder:z_batch})
        writer.add_summary(summary,i)

    if i%100==0:
        print ('Iteration:',i,'at',datetime.datetime.now())
        z_batch = np.random.normal(0,1,size=[batch_size,z_dimension])
        generated_image = generator(z_placeholder,1,z_dimension, reuse=True)
        images = sess.run(generated_image,feed_dict={z_placeholder:z_batch})
        # plt.imshow(images[0].reshape([28,28]),cmap='Greys')
        # plt.savefig('Generated_images/'+str(i)+'.jpg')

        img = images[0].reshape([1,28,28,1])
        result = discriminator(x_placeholder)
        estimate = sess.run(result,feed_dict={x_placeholder:img})
        print ('Estimate:',estimate)
