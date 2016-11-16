import tensorflow as tf
import random

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 240, 320, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 4])
y_c = tf.placeholder(tf.float32, shape=[None, 24])

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
h_pool4 = max_pool_2x2(h_conv4)

W_conv5 = weight_variable([3, 3, 64, 128])
b_conv5 = bias_variable([128])
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)

h_pool5_flat = tf.reshape(h_pool5, [-1, 10*8*128])

W_fc1 = weight_variable([10 * 8 * 128, 512])
b_fc1 = bias_variable([512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([512, 512])
b_fc2 = bias_variable([512])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc4 = weight_variable([512, 4])
b_fc4 = bias_variable([4])
y_conv=tf.nn.relu(tf.matmul(h_fc2_drop, W_fc4) + b_fc4)

W_fc1_c = weight_variable([10 * 8 * 128, 512])
b_fc1_c = bias_variable([512])
h_fc1_c = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1_c) + b_fc1_c)
h_fc1_drop_c = tf.nn.dropout(h_fc1_c, keep_prob)

W_fc2_c = weight_variable([512, 512])
b_fc2_c = bias_variable([512])
h_fc2_c = tf.nn.relu(tf.matmul(h_fc1_drop_c, W_fc2_c) + b_fc2_c)
h_fc2_drop_c = tf.nn.dropout(h_fc2_c, keep_prob)

W_fc4_c = weight_variable([512, 24])
b_fc4_c = bias_variable([24])
y_conv_c=tf.nn.softmax(tf.matmul(h_fc2_drop_c, W_fc4_c) + b_fc4_c)


y_conv_gg = tf.transpose(y_conv)
y_conv_t = tf.to_float([tf.minimum(tf.gather(y_conv_gg, 0), tf.gather(y_conv_gg, 2)),tf.minimum(tf.gather(y_conv_gg, 1), tf.gather(y_conv_gg, 3)),tf.maximum(tf.gather(y_conv_gg, 0), tf.gather(y_conv_gg, 2)),tf.maximum(tf.gather(y_conv_gg, 1), tf.gather(y_conv_gg, 3))])

y_t = tf.transpose(y_)

area_ground = tf.mul((tf.gather(y_t, 2) - tf.gather(y_t, 0)), (tf.gather(y_t, 3) - tf.gather(y_t, 1)))
area_predicted = tf.mul((tf.gather(y_conv_t, 2) - tf.gather(y_conv_t, 0)), (tf.gather(y_conv_t, 3) - tf.gather(y_conv_t, 1)))
overlap = tf.mul(tf.maximum(0.0, tf.sub(tf.minimum(tf.gather(y_t, 2), tf.gather(y_conv_t, 2)),tf.maximum(tf.gather(y_t, 0), tf.gather(y_conv_t, 0)))) , tf.maximum(0.0, tf.sub(tf.minimum(tf.gather(y_t, 3), tf.gather(y_conv_t, 3)),tf.maximum(tf.gather(y_t, 1), tf.gather(y_conv_t, 1)))))
union = tf.sub(tf.add(area_ground, area_predicted),overlap)
iou = tf.truediv(overlap, union)

with tf.name_scope('loss'):
    # loss = tf.reduce_mean(tf.sub(1.0,iou))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_c*tf.log(tf.clip_by_value(y_conv_c,1e-10,1.0)), reduction_indices=[1]))
    regression_loss = tf.nn.l2_loss(y_conv-y_)
    loss = 0.1*regression_loss + 10000*cross_entropy
    tf.scalar_summary('error', loss)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
train_step_r = tf.train.AdamOptimizer(learning_rate).minimize(regression_loss)
train_step_c = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, var_list=[W_fc1_c, b_fc1_c, h_fc1_c, h_fc1_drop_c, W_fc2_c, b_fc2_c, h_fc2_c, h_fc2_drop_c, W_fc4_c, b_fc4_c, y_conv_c])


with tf.name_scope('accuracy'):
    correct_prediction_r = tf.round(iou)
    accuracy_r = tf.reduce_mean(tf.cast(correct_prediction_r, tf.float32))

    correct_prediction_c = tf.equal(tf.argmax(y_conv_c,1), tf.argmax(y_c,1))
    accuracy_c = tf.reduce_mean(tf.cast(correct_prediction_c, tf.float32))

    # acc_summary = tf.scalar_summary('accuracy', accurvalidation)


merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('./train', sess.graph)
test_writer = tf.train.SummaryWriter('./test')

sess.run(tf.initialize_all_variables())

def start_t1(train_data, train_boxes, train_labels, validation_data, validation_boxes, validation_labels, test_data, test_boxes, test_labels):
    train_tuple = zip(train_data, train_boxes, train_labels)

    for i in range(30000):
        batch = random.sample(train_tuple, 32)
        batch_data = [zz[0] for zz in batch]
        batch_boxes = [zz[1] for zz in batch]
        batch_labels = [zz[2] for zz in batch]


        if i%500==0:
            va = 0
            for j in xrange(0, len(train_data), 8):
                mx = min(j+8, len(train_data))
                va = va + (accuracy_r.eval(feed_dict={x: train_data[j:mx], y_: train_boxes[j:mx], keep_prob: 1.0}))*(mx-j)

            va /= len(train_data)
            print "train localization", va

            va = 0
            for j in xrange(0, len(train_data), 8):
                mx = min(j+8, len(train_data))
                va = va + (accuracy_c.eval(feed_dict={x: train_data[j:mx], y_c: train_labels[j:mx], keep_prob: 1.0}))*(mx-j)

            va /= len(train_data)
            print "train classification", va


            va = 0
            for j in xrange(0, len(validation_data), 8):
                mx = min(j+8, len(validation_data))
                va = va + (accuracy_r.eval(feed_dict={x: validation_data[j:mx], y_: validation_boxes[j:mx], keep_prob: 1.0}))*(mx-j)

            va /= len(validation_data)
            print "validation localization", va

            va = 0
            for j in xrange(0, len(validation_data), 8):
                mx = min(j+8, len(validation_data))
                va = va + (accuracy_c.eval(feed_dict={x: validation_data[j:mx], y_c: validation_labels[j:mx], keep_prob: 1.0}))*(mx-j)

            va /= len(validation_data)
            print "validation classification", va

        if i%10 == 0 and i!=0:
            print "step", i, "loss", loss_val

        if i<10000:
            # _, loss_val, summary, conv_out, xx,cc,vv,bb,mm = sess.run([train_step_r, regression_loss, merged, y_conv_t, area_predicted, area_ground, overlap, union, iou], feed_dict={x:batch_data, y_: batch_boxes, keep_prob: 0.5, learning_rate: 1e-3})
            _, loss_val, summary  = sess.run([train_step_r, regression_loss, merged], feed_dict={x:batch_data, y_: batch_boxes, keep_prob: 0.5, learning_rate: 1e-3})

        else:
            _, loss_val, summary = sess.run([train_step_c, loss, merged], feed_dict={x:batch_data, y_c: batch_labels, keep_prob: 0.5, learning_rate: 1e-4})

        # if i % 1000 ==0:
        #     print "conv_predicted", conv_out
        #     print "overlap", vv
        #     print "iou", mm

        if i>100:
            train_writer.add_summary(summary, i)

    # import numpy as np
    # output = np.zeros((1,4,1))
    # # va = 0
    # for j in xrange(0, len(test_data), 8):
    #     mx = min(j+8, len(test_data))
    #     va = va + (accuracy.eval(feed_dict={x: test_data[j:mx], y_: test_boxes[j:mx], y_c: test_labels[j:mx], keep_prob: 1.0}))*(mx-j)
    #     # qq = sess.run([y_conv_t], feed_dict={x: test_data[j:mx], y_: test_labels[j:mx], keep_prob: 1.0})
    #     # output = np.vstack((output, np.transpose(qq)))
    #
    # va /= len(test_data)
    # print va
    # print output[1:]
    # np.save("output.txt", output[1:])




    va = 0
    for j in xrange(0, len(test_data), 8):
        mx = min(j+8, len(test_data))
        va = va + (accuracy_r.eval(feed_dict={x: test_data[j:mx], y_: test_boxes[j:mx], keep_prob: 1.0}))*(mx-j)

    va /= len(test_data)
    print "test localization", va

    va = 0
    for j in xrange(0, len(test_data), 8):
        mx = min(j+8, len(test_data))
        va = va + (accuracy_c.eval(feed_dict={x: test_data[j:mx], y_c: test_labels[j:mx], keep_prob: 1.0}))*(mx-j)

    va /= len(test_data)
    print "test classification", va
