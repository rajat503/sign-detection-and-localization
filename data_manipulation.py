import csv
from skimage import io
import os

train_data = []
train_labels = []
validation_data = []
validation_labels = []
test_data = []
test_labels = []
with open('trial_data/bounding_boxes.csv', 'rb') as csvfile:
    x=csv.reader(csvfile)
    for row in x:
        if row[0]=='file_name':
            continue
        label = row[5]
        image = io.imread('trial_data/cropped/'+label+'/'+row[0][:-7]+'crop'+'.jpg')
        data_vector = image.flatten()
        user_id = int((row[0].split('_'))[0])
        if user_id > 16:
            test_data.append(data_vector)
            test_labels.append(label)
        else:
            if user_id >12:
                validation_data.append(data_vector)
                validation_labels.append(label)
            else:
                train_data.append(data_vector)
                train_labels.append(label)
