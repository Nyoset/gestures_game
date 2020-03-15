from tensorflow import keras, losses
import matplotlib.pyplot as plt
import numpy as np

from os import environ
import os
import os.path as path
import argparse


dir_path = path.dirname(path.realpath(__file__))

save_path = path.join(dir_path, os.pardir, "model")
data_path = path.join(dir_path, os.pardir, "shapes")
image_size = 60


def show_grayscale_image(image):
    plt.figure()
    plt.imshow(image[:, :, 0], cmap='gray', vmin=0, vmax=1)
    plt.show()


def read_data():
    train_image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                         zoom_range=0.2,
                                                                         rotation_range=5,
                                                                         horizontal_flip=True,
                                                                         #preprocessing_function=remove_channels
                                                                         )

    train_data = train_image_generator.flow_from_directory(batch_size=6,
                                                           directory=data_path,
                                                           color_mode='grayscale',
                                                           shuffle=True,
                                                           target_size=(image_size, image_size))

    #x, y = train_data.next()
    #for sh in x:
        #show_grayscale_image(sh)

    return train_data


def train(training_data):
    model = keras.Sequential([
        keras.layers.Dense(units=256, activation='relu', input_shape=(image_size, image_size, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(
        training_data
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
