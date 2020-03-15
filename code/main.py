from tensorflow import keras, losses
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from os import listdir, scandir, environ
import os
import os.path as path
import argparse

import imageio
import glob

dir_path = path.dirname(path.realpath(__file__))

save_path = path.join(dir_path, os.pardir, "model")
data_path = path.join(dir_path, os.pardir, "shapes")
image_size = 24


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.shw()


def trim_label(label):
    return label.split(path.sep)[-1]


def saturate(pixel):
    return np.mean(pixel[0:2]) / 255


def remove_channels(image):
    return np.asarray([[saturate(rgba) for rgba in row] for row in image])


def read_data():
    shapes_data = []
    categories_data = []
    categories = [f.path for f in scandir(data_path) if f.is_dir()]

    train_image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                         zoom_range=0.2,
                                                                         rotation_range=5,
                                                                         horizontal_flip=True)

    train_data = train_image_generator.flow_from_directory(batch_size=2,
                                                           directory=data_path,
                                                           shuffle=True,
                                                           target_size=(image_size, image_size))

    return train_data


def encode(name):
    return 0 if name == 'circle' else 1


def train(data):
    model = keras.Sequential([
        keras.layers.Dense(units=256, activation='relu', input_shape=(image_size, image_size, 3)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(2, 2, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit_generator(
        data
    )

    model.save(save_path)
    model.summary()


def predict():
    model = keras.models.load_model(save_path)
    model.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TODO: Write HELP')

    parser.add_argument('-t', '--train', dest='train', action='store_true')
    args = parser.parse_args()

    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if args.train:
        data = read_data()
        train(data)
    else:
        predict()
