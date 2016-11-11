import csv
from skimage import io
import os
import numpy as np
from localization import start_t1
import sys
from skimage.viewer import ImageViewer

def train(user_list):
    train_data = []
    train_labels = []
    train_boxes =[]
    validation_data = []
    validation_labels = []
    validation_boxes = []
    test_data = []
    test_labels = []
    test_boxes = []

    for user in user_list:
        with open('../dataset/'+user+'/'+user+'_loc.csv', 'rb') as csvfile:
            x=csv.reader(csvfile)
            for row in x:
                if row[0]=='image':
                    continue
                image = io.imread('../dataset/'+row[0])
                data_vector = image
                # data_vector = np.array(image.flatten()).tolist()
                # sys.exit(0)
                ground_truth = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]

                user_id = int(user.split('_')[1])
                if user_id > 16:
                    test_data.append(data_vector)
                    test_boxes.append(ground_truth)
                else:
                    if user_id >13:
                        validation_data.append(data_vector)
                        validation_boxes.append(ground_truth)
                    else:
                        train_data.append(data_vector)
                        train_boxes.append(ground_truth)


    start_t1(train_data, train_boxes, validation_data, validation_boxes, test_data, test_boxes)
# train(['user_3','user_4','user_5','user_6','user_7','user_9','user_10','user_11','user_12','user_13','user_14','user_15','user_16','user_17','user_18','user_19'])
train(['user_3','user_4', 'user_5','user_6','user_7','user_9','user_10', 'user_11', 'user_12', 'user_13', 'user_14' ,'user_15', 'user_16', 'user_17', 'user_18','user_19'])
