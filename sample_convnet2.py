from keras.applications import vgg19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import ssl
import os
import glob 

ssl._create_default_https_context = ssl._create_unverified_context


#Load Model
vgg_model = vgg19.VGG19(weights='imagenet')
 
imgpath = '/Users/tuomastalvitie/documents/Images_copy/Clothing'

def import_image(imgpath):
	# load an image in PIL format
	original = load_img(imgpath, target_size=(224, 224))
	print('PIL image size',original.size)
	plt.imshow(original)
	plt.show()
	# convert the PIL image to a numpy array
	# IN PIL - image is in (width, height, channel)
	# In Numpy - image is in (height, width, channel)
	numpy_image = img_to_array(original)
	plt.imshow(np.uint8(numpy_image))
	plt.show()
	print('numpy array size',numpy_image.shape)
	return numpy_image



for i in glob.glob('/Users/tuomastalvitie/documents/Images_copy/Clothing/*.jpg'):
	image = import_image(os.path.join(imgpath, i))
	processed_image = vgg19.preprocess_input(image.copy())
	plt.imshow(processed_image)
	plt.show()
	processed_image = np.expand_dims(processed_image, axis=0)
	predictions = vgg_model.predict(processed_image)
	label = decode_predictions(predictions)
	print label
	# Convert the image / images into batch format
	# expand_dims will add an extra dimension to the data at a particular axis
	# We want the input matrix to the network to be of the form (batchsize, height, width, channels)
	# Thus we add the extra dimension to the axis 0.
# image_batch = np.expand_dims(numpy_image, axis=0)
# print('image batch size', image_batch.shape)
# plt.imshow(np.uint8(image_batch[0]))

# processed_image = vgg19.preprocess_input(image_batch.copy())
# plt.imshow(processed_image[0])
# plt.show()

# # get the predicted probabilities for each class
# predictions = vgg_model.predict(processed_image)
# # print predictions
 
# # convert the probabilities to class labels
# # We will get top 5 predictions which is the default
# label = decode_predictions(predictions)
# print label




