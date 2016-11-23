import csv
from skimage import io
import os
import numpy as np
import localization
import sys

def train(user_list, path):
    train_data = []
    train_boxes =[]

    for user in user_list:
        with open(path+user+'/'+user+'_loc.csv', 'rb') as csvfile:
            x=csv.reader(csvfile)
            for row in x:
                if row[0]=='image':
                    continue
                image = io.imread(path+row[0])
                data_vector = image
                # data_vector = np.array(image.flatten()).tolist()
                # sys.exit(0)
                ground_truth = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]

                user_id = int(user.split('_')[1])
                train_data.append(data_vector)
                train_boxes.append(ground_truth)


    localization.train(train_data, train_boxes)
# train(['user_3','user_4','user_5','user_6','user_7','user_9','user_10','user_11','user_12','user_13','user_14','user_15','user_16','user_17','user_18','user_19'])
# train(['user_3','user_4', 'user_5','user_6','user_7','user_9','user_10', 'user_11', 'user_12', 'user_13', 'user_14' ,'user_15', 'user_16', 'user_17', 'user_18','user_19'])
