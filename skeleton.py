import wrapper
from skimage import io
from skimage import transform
import os
import numpy as np

alphabet = "ABCDEFGHIKLMNOPQRSTUVWXY"

class GestureRecognizer(object):

	"""class to perform gesture recognition"""

	path = ""

	def __init__(self, data_directory):

		"""
			data_directory : path like /home/sanket/mlproj/dataset/
			includes the dataset folder with '/'

			Initialize all your variables here
		"""
		self.path = data_directory

	def train(self, train_list):

		"""
			train_list : list of users to use for training
			eg ["user_1", "user_2", "user_3"]

			The train function should train all your classifiers,
			both binary and multiclass on the given list of users
		"""
		wrapper.train_classifier(train_list, self.path)
		wrapper.train_localizer(train_list, self.path)



	def recognize_gesture(self, image):

		"""
			image : a 320x240 pixel RGB image in the form of a numpy array

			This function should locate the hand and classify the gesture.

			returns : (position, label)

			position : a tuple of (x1,y1,x2,y2) coordinates of bounding box
					   x1,y1 is top left corner, x2,y2 is bottom right

			label : a single character. eg 'A' or 'B'
		"""

		x1, y1, x2, y2 = wrapper.test_localizer(image)
		conv_output = np.array(wrapper.test_classifier(image, x1, y1, x2, y2)).tolist()[0][0]
		conv_tuples = [(conv_output[i], alphabet[i]) for i in xrange(len(conv_output))]
		conv_tuples.sort(reverse=True)
		labels = [conv_tuples[i][1] for i in xrange(5)]
		# print conv_tuples
		# print labels
		return (x1, y1, x2, y2), labels

	# def translate_video(self, image_array):
	#
	# 	"""
	# 		image_array : a list of images as described above.
	# 					  can be of arbitrary length
	#
	# 		This function classifies the video into a 5 character string
	#
	# 		returns : word (a string of 5 characters)
	# 				  no two consecutive characters are identical
	# 	"""
	#
	# 	return word

obj = GestureRecognizer('dataset/')
obj.train(['user_3','user_4', 'user_5','user_6','user_7','user_9','user_10', 'user_11', 'user_12', 'user_13'])
# obj.recognize_gesture(['user_17','user_18', 'user_19'])
image = io.imread('dataset/user_17/A0.jpg')
print obj.recognize_gesture(image)
