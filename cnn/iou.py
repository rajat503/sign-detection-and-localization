import numpy as np
import csv
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def area(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy

def iou(A, B):
	ra = Rectangle(*A)
	rb = Rectangle(*B)
	intersection = area(ra, rb) if area(ra, rb) != None else 0
	area_a = (ra.xmax-ra.xmin)*(ra.ymax-ra.ymin)
	area_b = (rb.xmax-rb.xmin)*(rb.ymax-rb.ymin)
	union  = area_a + area_b - intersection
	return 1.0*intersection/union  # WARNING: union = 0  case not handled

# L1 = np.array([[[1],[2],[3],[4]], [[1],[2],[3],[4]]])
# L2 = [[1,2,3,4], [1,2,3,100]]

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
                ground_truth = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]
                user_id = int(user.split('_')[1])
                if user_id > 16:
                    test_boxes.append(ground_truth)

    return test_boxes

L1 = np.load("output.txt.npy")
L1 = L1.astype(int)
N = len(L1)
L2 = train(['user_3','user_4', 'user_5','user_6','user_7','user_9','user_10', 'user_11', 'user_12', 'user_13', 'user_14' ,'user_15', 'user_16', 'user_17', 'user_18','user_19'])

c = 0
for i in xrange(N):
	if iou(L1[i].flatten(), L2[i]) > 0.5:
		c += 1

print 1.0*c/N
