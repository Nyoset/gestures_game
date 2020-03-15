from tensorflow import keras, losses
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from os import listdir, scandir, environ
import os
import os.path as path

import imageio
import glob

dir_path = path.dirname(path.realpath(__file__))

data_path = path.join(dir_path, os.pardir, "shapes")
image_size = 120


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

    for category in categories:
        for im_path in glob.glob(path.join(data_path, category, '*.png')):
            image = imageio.imread(im_path)
            saturated_image = remove_channels(image)
            shapes_data.append(saturated_image)
            categories_data.append(encode(trim_label(category)))

    return np.array(categories_data), np.array(shapes_data)


def encode(name):
    return 0 if name == 'circle' else 1


def train(labels, images):
    model = keras.Sequential([
        keras.layers.Dense(units=256, activation='relu', input_shape=(image_size, image_size)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(
        images,
        keras.utils.to_categorical(labels)
    )

    print(model.predict(images, batch_size=64)[0])


if __name__ == "__main__":
    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    labels, images = read_data()
    train(labels, images)
