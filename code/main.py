from tensorflow import keras, losses
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from os import listdir, scandir, environ
from os.path import join, sep

import imageio
import glob


data_path = "/Users/marcbasquens/Desktop/gesture/shapes"
image_size = 120


def show_image(image):
	plt.figure()
	plt.imshow(image) 
	plt.show()

def trim_label(label):
	return label.split(sep)[-1]

def saturate(pixel):
	return np.mean(pixel[0:2]) / 255

def remove_channels(image):
	return np.asarray([[saturate(rgba) for rgba in row] for row in image])

def read_data():
	shapes_data = np.empty([image_size, image_size])
	categories_data = np.asarray([])
	categories = [f.path for f in scandir(data_path) if f.is_dir()]

	for category in categories:
		for im_path in glob.glob(join(data_path, category, '*.png')):			
			image = imageio.imread(im_path)
			saturated_image = remove_channels(image))
			np.concatenate(shapes_data, saturated_image)
			np.concatenate(categories_data, trim_label(category))

	return categories_data, shapes_data

def encode(str):
	return 0 if str == 'circle' else 1

def train(labels, images):
	model = keras.Sequential([
	    keras.layers.Reshape(target_shape=(image_size * image_size,), input_shape=(image_size, image_size)),
	    keras.layers.Dense(units=256, activation='relu'),
	    keras.layers.Dense(units=2, activation='softmax')
	])

	model.compile(optimizer='adam', 
              loss=losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

	history = model.fit(
	    labels, 
	    images
	)

if __name__== "__main__":
	environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	labels, images = read_data()
	train(labels, images)
