import csv
from skimage import io
import os
import numpy as np
from classification import start_t1
import sys

def train(user_list):
    train_data = []
    train_labels = []
    validation_data = []
    validation_labels = []
    test_data = []
    test_labels = []

    for user in user_list:
        with open('../dataset/'+user+'/'+user+'_loc.csv', 'rb') as csvfile:
            x=csv.reader(csvfile)
            for row in x:
                if row[0]=='image':
                    continue
                image = io.imread('../dataset/'+row[0])
                data_vector = image
                one_hot = [0 for i in range(24)]
                letter = (row[0].split('/')[1])[0]
                label = ord(letter)-65
                if letter > 'J':
                    label = label - 1
                one_hot[int(label)]=1
                user_id = int(user.split('_')[1])
                if user_id > 16:
                    test_data.append(data_vector)
                    test_labels.append(one_hot)
                else:
                    if user_id >13:
                        validation_data.append(data_vector)
                        validation_labels.append(one_hot)
                    else:
                        train_data.append(data_vector)
                        train_labels.append(one_hot)


    start_t1(train_data, train_labels, validation_data, validation_labels, test_data, test_labels)

train(['user_3','user_4', 'user_5','user_6','user_7','user_9','user_10', 'user_11', 'user_12', 'user_13', 'user_14' ,'user_15', 'user_16', 'user_17', 'user_18','user_19'])
