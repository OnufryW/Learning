from __future__ import print_function
import random
import numpy as np
import tensorflow as tf
from six.moves import range
import matplotlib.pyplot as plt
import tools
import sys

filename = "DefaultGraph" if len(sys.argv) == 1 else sys.argv[1]

graph = tf.Graph()
with graph.as_default():
  # Input data.
  input_dataset = tf.placeholder(tf.float32, name="Input")
 
  # Variables.
  layer_width = 1000
  weights_1 = tf.Variable(tf.truncated_normal([28 * 28, layer_width]))
  weights_2 = tf.Variable(tf.truncated_normal([layer_width, tools.NUMLABELS]))
  biases_1 = tf.Variable(tf.zeros([layer_width]))
  biases_2 = tf.Variable(tf.zeros([tools.NUMLABELS]))
 
  def outputs(inputs, dropout=False):
    level_1 = tf.matmul(inputs, weights_1) + biases_1
    after_neuron = tf.nn.relu(level_1)
    if dropout:
      return tf.matmul(tf.nn.dropout(after_neuron, 0.8), weights_2) + biases_2
    return tf.matmul(after_neuron, weights_2) + biases_2

  # Training computation.
  beta = 0.0
  logits = outputs(input_dataset, dropout=True)
  tf_train_labels = tf.placeholder(tf.float32, name="TrainLabels")
  loss = tf.add(tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)),
    beta * (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)), name="Loss")
  
  # Optimizer.
  optimizer = tf.train.MomentumOptimizer(0.3, 0.2).minimize(loss, name="Opt")
 
  # Prediction, which differs, because it doesn't drop out.
  prediction = tf.nn.softmax(outputs(input_dataset), name="Pred")
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  tf.train.export_meta_graph(filename=(filename + '.meta'), as_text=True)
  tools.SaveConfig(filename, ["Flat", "OneHot"])

with tf.Session(graph=graph) as session:
  saver = tf.train.Saver(tf.all_variables())
  tf.initialize_all_variables().run()
  print('Initialized')
  saver.save(session, filename)
