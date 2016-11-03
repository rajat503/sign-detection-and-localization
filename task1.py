import tensorflow as tf
import random

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 128*128*3])
y_ = tf.placeholder(tf.float32, shape=[None, 5])
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1,128,128,3])

W_conv1 = weight_variable([3, 3, 3, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 16, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

W_conv3 = weight_variable([3, 3, 16, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

W_conv4 = weight_variable([3, 3, 32, 64])
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

h_pool4_flat = tf.reshape(h_pool4, [-1, 32*32*64])

W_fc1 = weight_variable([32 * 32 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 1024])
b_fc2 = bias_variable([1024])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([1024, 1024])
b_fc3 = bias_variable([1024])
h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

W_fc4 = weight_variable([1024, 5])
b_fc4 = bias_variable([5])
y_conv=tf.nn.softmax(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), reduction_indices=[1]))
    tf.scalar_summary('cross entropy', cross_entropy)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_summary = tf.scalar_summary('accuracy', accuracy)


merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('./train', sess.graph)
test_writer = tf.train.SummaryWriter('./test')

sess.run(tf.initialize_all_variables())

def start_t1(train_data, train_labels, validation_data, validation_labels, test_data, test_labels):
    train_tuple = zip(train_data, train_labels)

    for i in range(5000):

        batch = random.sample(train_tuple, 16)
        batch_data = [zz[0] for zz in batch]
        batch_labels = [zz[1] for zz in batch]

        if i%100==0:
            summ = sess.run(acc_summary, feed_dict={x: validation_data, y_: validation_labels, keep_prob: 1.0})
            test_writer.add_summary(summ, i)
            print("train accuracy %g"%accuracy.eval(feed_dict={x: train_data, y_: train_labels, keep_prob: 1.0}))
            print("validation accuracy %g"%accuracy.eval(feed_dict={x: validation_data, y_: validation_labels, keep_prob: 1.0}))


        if i%10 == 0 and i!=0:
            print "step", i, "loss", loss_val

        if i < 2000:
            _, loss_val, summary = sess.run([train_step, cross_entropy, merged], feed_dict={x:batch_data, y_: batch_labels, keep_prob: 0.5, learning_rate: 1e-4})
        else:
            if i < 4000:
                _, loss_val, summary = sess.run([train_step, cross_entropy, merged], feed_dict={x:batch_data, y_: batch_labels, keep_prob: 0.5, learning_rate: 1e-5})
            else:
                _, loss_val, summary = sess.run([train_step, cross_entropy, merged], feed_dict={x:batch_data, y_: batch_labels, keep_prob: 0.5, learning_rate: 1e-6})

        if i>100:
            train_writer.add_summary(summary, i)


    print("test accuracy %g"%accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))

    det, pred = sess.run([tf.argmax(y_conv,1), correct_prediction], feed_dict={x:  validation_data, y_: validation_labels, keep_prob: 1.0})

    det1, pred1 = sess.run([tf.argmax(y_conv,1), correct_prediction], feed_dict={x:  test_data, y_: test_labels, keep_prob: 1.0})

    print "Validation faults"
    from skimage import io
    import numpy as np
    for i in range(len(pred)):
        if pred[i]==False:
            zx=np.array(validation_data[i])
            zx=zx.reshape(128,128,3)
            print "Predicted class", det[i]
            print "Correct class", validation_labels[i]
            img = io.imshow(zx)
            io.show()

    print "Test faults"
    for i in range(len(pred1)):
        if pred1[i]==False:
            zx=np.array(test_data[i])
            zx=zx.reshape(128,128,3)
            print "Predicted class", det1[i]
            print "Correct class", test_labels[i]
            img = io.imshow(zx)
            io.show()
