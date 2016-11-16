import csv
from skimage import io
from skimage import transform
import os
import numpy as np
import classify_top5
import sys

def train(user_list, path):
    train_data = []
    train_labels = []

    for user in user_list:
        with open(path+user+'/'+user+'_loc.csv', 'rb') as csvfile:
            x=csv.reader(csvfile)
            for row in x:
                if row[0]=='image':
                    continue
                user_id = int(user.split('_')[1])
                image = io.imread(path+row[0])

                data_vector1 = image[int(row[2]):int(row[4]),int(row[1]):int(row[3])]

                data_vector = transform.resize(data_vector1, (128, 128))

                one_hot = [0 for i in range(24)]
                letter = (row[0].split('/')[1])[0]
                label = ord(letter)-65
                if letter > 'J':
                    label = label - 1
                one_hot[int(label)]=1

                train_data.append(data_vector)
                train_labels.append(one_hot)


    classify_top5.train(train_data, train_labels)


def test(image, x1, y1, x2, y2):
    cropped_image = image[y1:y2,x1:x2]
    transformed_image = transform.resize(cropped_image, (128, 128))
    return classify_top5.test(transformed_image)

# train(['user_3','user_4', 'user_5','user_6','user_7','user_9','user_10', 'user_11', 'user_12', 'user_13', 'user_14' ,'user_15', 'user_16', 'user_17', 'user_18','user_19'])
