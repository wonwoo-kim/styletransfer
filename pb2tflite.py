from tensorflow.python.framework import graph_util
import tensorflow as tf

graph_def_file = "output_graph.pb"
input_arrays = ["img_placeholder"]
output_arrays = ["add_37"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("ts.tflite", "wb").write(tflite_model)
