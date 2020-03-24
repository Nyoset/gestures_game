from __future__ import print_function
# Keras' "get_session" function gives us easy access to the session where we train the graph
from keras import backend as K

import tensorflow as tf
# freeze_graph "screenshots" the graph
from tensorflow.python.tools import freeze_graph
# optimize_for_inference lib optimizes this frozen graph
from tensorflow.python.tools import optimize_for_inference_lib

# os and os.path are used to create the output file where we save our frozen graphs
import os
import os.path as path


# EXPORT GAPH FOR UNITY
def export_model(saver, input_node_names, output_node_name):
    # creates the 'out' folder where our frozen graphs will be saved
    if not path.exists('out'):
        os.mkdir('out')

    # an arbitrary name for our graph
    GRAPH_NAME = 'my_graph_name'

    # GRAPH SAVING - '.pbtxt'
    tf.train.write_graph(K.get_session().graph_def, 'out', GRAPH_NAME + '_graph.pbtxt')

    # GRAPH SAVING - '.chkp'
    # KEY: This method saves the graph at it's last checkpoint (hence '.chkp')
    saver.save(K.get_session(), 'out/' + GRAPH_NAME + '.chkp')

    # GRAPH SAVING - '.bytes'
    # freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
    # input_binary, checkpoint_path, output_node_names,
    # restore_op_name, filename_tensor_name,
    # output_frozen_graph_name, clear_devices, "")
    freeze_graph.freeze_graph('out/' + GRAPH_NAME + '_graph.pbtxt', None, False,
                              'out/' + GRAPH_NAME + '.chkp', output_node_name,
                              "save/restore_all", "save/Const:0",
                              'out/frozen_' + GRAPH_NAME + '.bytes', True, "")

    # GRAPH OPTIMIZING
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + GRAPH_NAME + '.bytes', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + GRAPH_NAME + '.bytes', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


# IN MAIN WITHIN TF SESSION SCOPE
export_model(tf.train.Saver(), ["input_node"], "output_node")