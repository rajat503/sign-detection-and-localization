import csv
from skimage import io
from skimage import transform
import os
import numpy as np
from crop_classify import start_t1
import sys
from skimage.viewer import ImageViewer

def train(user_list):
    train_data = []
    train_labels = []
    validation_data = []
    validation_labels = []
    test_data = []
    test_labels = []

    predicted_boxes = np.load("output.txt.npy")
    predicted_boxes = predicted_boxes.astype(int)

    count = 0
    for user in user_list:
        with open('../dataset/'+user+'/'+user+'_loc.csv', 'rb') as csvfile:
            x=csv.reader(csvfile)
            for row in x:
                if row[0]=='image':
                    continue
                user_id = int(user.split('_')[1])
                image = io.imread('../dataset/'+row[0])

                if user_id > 16:
                    data_vector1 = image[predicted_boxes[count][1][0]:predicted_boxes[count][3][0],predicted_boxes[count][0][0]:predicted_boxes[count][2][0]]
                    count = count + 1
                else:
                    data_vector1 = image[int(row[2]):int(row[4]),int(row[1]):int(row[3])]

                data_vector = transform.resize(data_vector1, (128, 128))

                one_hot = [0 for i in range(24)]
                letter = (row[0].split('/')[1])[0]
                label = ord(letter)-65
                if letter > 'J':
                    label = label - 1
                one_hot[int(label)]=1
                if user_id > 16:
                    test_data.append(data_vector)
                    test_labels.append(one_hot)
                else:
                    if user_id >14:
                        validation_data.append(data_vector)
                        validation_labels.append(one_hot)
                    else:
                        train_data.append(data_vector)
                        train_labels.append(one_hot)


    start_t1(train_data, train_labels, validation_data, validation_labels, test_data, test_labels)

train(['user_3','user_4', 'user_5','user_6','user_7','user_9','user_10', 'user_11', 'user_12', 'user_13', 'user_14' ,'user_15', 'user_16', 'user_17', 'user_18','user_19'])
