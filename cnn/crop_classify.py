import tensorflow as tf
import random

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 24])
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

W_conv1 = weight_variable([3, 3, 3, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 16, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([3, 3, 16, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([3, 3, 32, 64])
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

W_conv5 = weight_variable([3, 3, 64, 128])
b_conv5 = bias_variable([128])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)

h_pool5_flat = tf.reshape(h_pool5, [-1, 8*8*128])

W_fc1 = weight_variable([8 * 8 * 128, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 1024])
b_fc2 = bias_variable([1024])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc4 = weight_variable([1024, 24])
b_fc4 = bias_variable([24])
y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc4) + b_fc4)

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

    for i in range(15000):

        batch = random.sample(train_tuple, 64)
        batch_data = [zz[0] for zz in batch]
        batch_labels = [zz[1] for zz in batch]

        if i%500==0:
            # test_writer.add_summary(summ, i)
            va = 0
            for j in xrange(0, len(train_data), 8):
                mx = min(j+8, len(train_data))
                va = va + (accuracy.eval(feed_dict={x: train_data[j:mx], y_: train_labels[j:mx], keep_prob: 1.0}))*(mx-j)

            va /= len(train_data)
            print "train", va

            va = 0
            for j in xrange(0, len(validation_data), 8):
                mx = min(j+8, len(validation_data))
                va = va + (accuracy.eval(feed_dict={x: validation_data[j:mx], y_: validation_labels[j:mx], keep_prob: 1.0}))*(mx-j)

            va /= len(validation_data)
            print "validation", va

        if i%10 == 0 and i!=0:
            print "step", i, "loss", loss_val

        if i<10000:
            _, loss_val, summary = sess.run([train_step, cross_entropy, merged], feed_dict={x:batch_data, y_: batch_labels, keep_prob: 0.5, learning_rate: 1e-4})
        else:
            _, loss_val, summary = sess.run([train_step, cross_entropy, merged], feed_dict={x:batch_data, y_: batch_labels, keep_prob: 0.5, learning_rate: 1e-5})


        if i>100:
            train_writer.add_summary(summary, i)

    va = 0
    for j in xrange(0, len(test_data), 8):
        mx = min(j+8, len(test_data))
        va = va + (accuracy.eval(feed_dict={x: test_data[j:mx], y_: test_labels[j:mx], keep_prob: 1.0}))*(mx-j)

    va /= len(test_data)
    print "test accurcy", va
