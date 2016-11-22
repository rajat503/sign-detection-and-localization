import tensorflow as tf
import random

# sess = tf.InteractiveSession()
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


g = tf.Graph()

with g.as_default():

    x = tf.placeholder(tf.float32, shape=[None, 240, 320, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 4])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

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


    y_conv_gg = tf.transpose(y_conv)
    y_conv_t = tf.to_float([tf.minimum(tf.gather(y_conv_gg, 0), tf.gather(y_conv_gg, 2)),tf.minimum(tf.gather(y_conv_gg, 1), tf.gather(y_conv_gg, 3)),tf.maximum(tf.gather(y_conv_gg, 0), tf.gather(y_conv_gg, 2)),tf.maximum(tf.gather(y_conv_gg, 1), tf.gather(y_conv_gg, 3))])

    y_t = tf.transpose(y_)

    area_ground = tf.mul((tf.gather(y_t, 2) - tf.gather(y_t, 0)), (tf.gather(y_t, 3) - tf.gather(y_t, 1)))
    area_predicted = tf.mul((tf.gather(y_conv_t, 2) - tf.gather(y_conv_t, 0)), (tf.gather(y_conv_t, 3) - tf.gather(y_conv_t, 1)))
    overlap = tf.mul(tf.maximum(0.0, tf.sub(tf.minimum(tf.gather(y_t, 2), tf.gather(y_conv_t, 2)),tf.maximum(tf.gather(y_t, 0), tf.gather(y_conv_t, 0)))) , tf.maximum(0.0, tf.sub(tf.minimum(tf.gather(y_t, 3), tf.gather(y_conv_t, 3)),tf.maximum(tf.gather(y_t, 1), tf.gather(y_conv_t, 1)))))
    union = tf.sub(tf.add(area_ground, area_predicted),overlap)
    iou = tf.truediv(overlap, union)

    # loss = tf.reduce_mean(tf.sub(1.0,iou))
    loss = tf.nn.l2_loss(y_conv-y_)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.round(iou)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    initialize = tf.initialize_all_variables()
    saver = tf.train.Saver()


sess = tf.Session(graph=g)
sess.run(initialize)

def train(train_data, train_labels):
    train_tuple = zip(train_data, train_labels)

    for i in range(1):

        batch = random.sample(train_tuple, 32)
        batch_data = [zz[0] for zz in batch]
        batch_labels = [zz[1] for zz in batch]

        with sess.as_default():
            if i%500==0:
                va = 0
                for j in xrange(0, len(train_data), 8):
                    mx = min(j+8, len(train_data))
                    va = va + (accuracy.eval(feed_dict={x: train_data[j:mx], y_: train_labels[j:mx], keep_prob: 1.0}))*(mx-j)

                va /= len(train_data)
                print "train", va

        if i%10 == 0 and i!=0:
            print "step", i, "loss", loss_val

        # if i<1500:
        _, loss_val, conv_out, xx,cc,vv,bb,mm = sess.run([train_step, loss, y_conv_t, area_predicted, area_ground, overlap, union, iou], feed_dict={x:batch_data, y_: batch_labels, keep_prob: 0.5, learning_rate: 1e-3})
        # else:
        #     _, loss_val, summary, conv_out, xx,cc,vv,bb,mm = sess.run([train_step, loss, merged, y_conv_t, area_predicted, area_ground, overlap, union, iou], feed_dict={x:batch_data, y_: batch_labels, keep_prob: 0.8, learning_rate: 1e-4})

        # if i % 1000 ==0:
        #     print "conv_predicted", conv_out
        #     print "overlap", vv
        #     print "iou", mm

def test(image):
    import numpy as np
    image_list = []
    image_list.append(image)
    conv_out = sess.run([y_conv_t], feed_dict={x: image_list, keep_prob: 1.0})
    predicted_box = np.array(conv_out).astype(int)
    return predicted_box[0]

def save(file_name):
    save_path = saver.save(sess, file_name)
    print save_path

def load(file_name):
    saver.restore(sess, file_name)
    print "Model restored"
