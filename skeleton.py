class GestureRecognizer(object):

	"""class to perform gesture recognition"""

	def __init__(self, data_directory):

		"""
			data_directory : path like /home/sanket/mlproj/dataset/
			includes the dataset folder with '/'

			Initialize all your variables here
		"""

	def train(self, train_list):

		"""
			train_list : list of users to use for training
			eg ["user_1", "user_2", "user_3"]

			The train function should train all your classifiers,
			both binary and multiclass on the given list of users
		"""

	def recognize_gesture(self, image):

		"""
			image : a 320x240 pixel RGB image in the form of a numpy array

			This function should locate the hand and classify the gesture.

			returns : (position, label)

			position : a tuple of (x1,y1,x2,y2) coordinates of bounding box
					   x1,y1 is top left corner, x2,y2 is bottom right

			label : a single character. eg 'A' or 'B'
		"""

		return position, label

	def translate_video(self, image_array):

		"""
			image_array : a list of images as described above.
						  can be of arbitrary length

			This function classifies the video into a 5 character string

			returns : word (a string of 5 characters)
					  no two consecutive characters are identical
		"""

		return word