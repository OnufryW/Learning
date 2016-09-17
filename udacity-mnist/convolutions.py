from __future__ import print_function
import numpy as np
import tensorflow as tf
import tools
import sys

filename = "DefaultGraph" if len(sys.argv) == 1 else sys.argv[1]

patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()
with graph.as_default():
  input_dataset = tf.placeholder(tf.float32, name="Input")

  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, 1, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal(
      [7 * 7 * depth, num_hidden], stddev = 0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, tools.NUMLABELS], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[tools.NUMLABELS]))

  tools.SaveConfig("Channel", "OneHot")

  conv = tf.nn.conv2d(input_dataset, layer1_weights, [1,1,1,1], padding='SAME')
  pool = tf.nn.max_pool(conv, [1,2,2,1], [1,2,2,1], padding='SAME')
  hidden = tf.nn.relu(pool + layer1_biases)
  conv = tf.nn.conv2d(hidden, layer2_weights, [1,1,1,1], padding='SAME')
  pool = tf.nn.max_pool(conv, [1,2,2,1], [1,2,2,1], padding='SAME')
  hidden = tf.nn.relu(pool + layer2_biases)
  reshape = tf.reshape(hidden, [-1, 7 * 7 * depth])
  hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
  logits = tf.matmul(hidden, layer4_weights) + layer4_biases
  tf_train_labels = tf.placeholder(tf.float32, name="TrainLabels")
  loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels),
          name="Loss")
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss, name="Opt")
  prediction = tf.nn.softmax(logits, name="Pred")
  tf.train.export_meta_graph(filename=(filename + '.meta'))
  tools.SaveConfig(filename, ["Channel", "OneHot"])

with tf.Session(graph=graph) as session:
  saver = tf.train.Saver(tf.all_variables())
  tf.initialize_all_variables().run()
  print('Initialized')
  saver.save(session, filename)

