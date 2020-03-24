from mlagents.tf_utils import tf
#import tensorflow as tf

from tensorflow.core.protobuf import saver_pb2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import convert_to_constants

from os import environ
import os
import os.path as path
import argparse

import mlagents as mla

here = path.dirname(path.realpath(__file__))

save_path = path.join(here, os.pardir, "model")
data_path = path.join(here, os.pardir, "shapes")
image_size = 10


def show_grayscale_image(image):
    plt.figure()
    plt.imshow(image[:, :, 0], cmap='gray', vmin=0, vmax=1)
    plt.show()


def read_data():
    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                         zoom_range=0.2,
                                                                         rotation_range=5,
                                                                         horizontal_flip=True,
                                                                         #preprocessing_function=remove_channels
                                                                         )

    train_data = train_image_generator.flow_from_directory(#batch_size=6,
                                                           directory=data_path,
                                                           color_mode='grayscale',
                                                           class_mode='categorical',
                                                           shuffle=True,
                                                           target_size=(image_size, image_size))

    #x, y = train_data.next()
    #for sh in x:
        #show_grayscale_image(sh)

    return train_data


def read_data_v1():

    pass


def train(training_data):

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(image_size, image_size, 1)),
        tf.keras.layers.Dense(units=20, activation='relu'),
        tf.keras.layers.Dense(units=2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history = model.fit(
        training_data
    )

    model.summary()
    #model.save(save_path)

    @tf.function(input_signature=[tf.TensorSpec(shape=[image_size * image_size, None], dtype=tf.float32)])
    def to_save(x):
        return model(x)

    f = to_save.get_concrete_function()
    constant_graph = convert_to_constants.convert_variables_to_constants_v2(f)
    tf.io.write_graph(constant_graph.graph.as_graph_def(), save_path, "constant_graph.pb", as_text=False)


def predict():
    model = tf.keras.models.load_model(save_path)
    model.summary()


def train_as_ml():
    with self.graph.as_default():
        last_checkpoint = self.model_path + "/model-" + str(steps) + ".ckpt"
        self.saver.save(self.sess, last_checkpoint)
        tf.train.write_graph(
            self.graph, self.model_path, "raw_graph_def.pb", as_text=False
        )
    pass


def get_model(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, image_size, image_size, 1])

    '''
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    '''
    flat = tf.reshape(input_layer, [-1, image_size * image_size])
    dense = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)
    #dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dense, units=2)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        model = tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)
        return model


def train_v1(data):

    train_data = data.next()[0]
    train_labels = data.labels

    with tf.Session() as sess:

        classifier = tf.estimator.Estimator(
            model_fn=get_model, model_dir=save_path)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=3,
            num_epochs=None,
            shuffle=True)


        # train one step and display the probabilties
        sess.run(
            classifier.train(input_fn=train_input_fn, steps=1)
        )

        file_writer = tf.summary.FileWriter(os.path.join(save_path, "writer"), sess.graph)


        '''freeze_graph.freeze_graph(
                                input_graph=os.path.join(save_path, "graph.pbtxt"),
                                input_saver=None,
                                input_binary=False,
                                input_checkpoint=os.path.join(save_path, "model.ckpt-14.index"),
                                output_node_names=["dense/MatMul"],
                                restore_op_name=None,
                                filename_tensor_name=None,
                                output_graph=None,
                                clear_devices=None,
                                initializer_nodes=None,
                                variable_names_whitelist="",
                                variable_names_blacklist="",
                                input_meta_graph=None,
                                input_saved_model_dir=None,
                                checkpoint_version=saver_pb2.SaverDef.V1
                                )'''

    '''
def _initialize_graph():
    with self.graph.as_default():
        self.saver = tf.train.Saver(max_to_keep=self.keep_checkpoints)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    policy = self.get_policy(name_behavior_id)
    settings = SerializationSettings(policy.model_path, policy.brain.brain_name)
    export_policy_model(settings, policy.graph, policy.sess)
    
    with self.graph.as_default():
    last_checkpoint = self.model_path + "/model-" + str(steps) + ".ckpt"
    self.saver.save(self.sess, last_checkpoint)
    tf.train.write_graph(
        self.graph, self.model_path, "raw_graph_def.pb", as_text=False
    )
'''



'''

knn_prediction = tf.reduce_sum(tf.abs(tf.add(train_pl, tf.negative(test_pl))), axis=1)

pred = tf.argmin(knn_prediction, 0)

with tf.Session() as tf_session:
    missed = 0

    for i in xrange(len(test_dataset)):
        knn_index = tf_session.run(pred, feed_dict={train_pl: train_dataset, test_pl: test_dataset[i]})

        print "Predicted class {} -- True class {}".format(train_labels[knn_index], test_labels[i])

        if train_labels[knn_index] != test_labels[i]:
            missed += 1

    tf.summary.FileWriter("../samples/article/logs", tf_session.graph)
'''

if __name__ == "__main__":

    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    data = read_data()
    train_v1(data)

