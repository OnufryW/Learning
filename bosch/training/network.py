from six.moves import range
import numpy as np
import tensorflow as tf
import sys
import random
import argparse

parser = argparse.ArgumentParser(description='Create and init the graph.')
parser.add_argument('-d', '--depth', default=2, type=int,
        help='Number of network layers')
parser.add_argument('-w', '--width', default=1024, type=int,
        help='Width of a single layer')
parser.add_argument('-k', '--keep_rate', default=0.9, type=float,
        help='Keep ratio of dropout layers')
parser.add_argument('filename', help='Filename for the graph')

args = parser.parse_args()

input_size=1936 
sdev=0.01

assert args.depth > 1
assert args.keep_rate > 0
assert args.keep_rate <= 1

with open(args.filename + ".log", "w") as logfile:
  logfile.write("%s %f %d %d\n" % (args.filename, args.keep_rate, args.depth, args.width))

seed = random.randint(0, 2 ** 30)
with open(args.filename + ".config", "w") as configfile:
  configfile.write(str(seed) + "\n")

graph = tf.Graph()
with graph.as_default():
  # Input data.
  input_dataset = tf.placeholder(tf.float32, name="Input")

  # Variables.
  weights = []
  biases = []

  weights.append(tf.Variable(
      tf.truncated_normal([input_size, args.width], stddev=sdev), name="weights1"))
  biases.append(tf.Variable(
      tf.truncated_normal([args.width], stddev=sdev), name="biases1"))
  for x in range(args.depth - 2):
    weights.append(tf.Variable(tf.truncated_normal(
        [args.width, args.width], stddev=sdev), name="weights"+str(x+2)))
    biases.append(tf.Variable(tf.truncated_normal(
        [args.width], stddev=sdev), name="biases"+str(x+2)))
  weights.append(tf.Variable(
      tf.truncated_normal([args.width, 1], stddev=sdev), name="weights"+str(args.depth)))
  biases.append(tf.Variable(
      tf.truncated_normal([1], stddev=sdev), name="biases"+str(args.depth)))

  # Let's do dropout on training to prevent overfitting.
  def construct_network(dropout):
    suffix = "D" if dropout else ""
    state = input_dataset
    for x in range(args.depth-1):
      operated = tf.matmul(
              state, weights[x], name="base"+str(x)+suffix) + biases[x]
      neuron = tf.nn.relu(operated, name="neuron"+str(x)+suffix)
      if dropout:
        state = tf.nn.dropout(neuron, args.keep_rate)
      else:
        state = neuron
    return tf.matmul(state, weights[-1], name="final"+suffix) + biases[-1]

  logits = tf.reshape(tf.tanh(construct_network(True)) / 2. + 0.5, [-1], name="Logits")
  tf_train_labels = tf.placeholder(tf.float32, name="TrainLabels")

  true_positive = tf_train_labels * logits
  true_negative = (1. - tf_train_labels) * (1. - logits)
  false_positive = (1. - tf_train_labels) * logits
  false_negative = tf_train_labels * (1. - logits)

  true_positives = tf.reduce_sum(true_positive, name="TruePos")
  true_negatives = tf.reduce_sum(true_negative, name="TrueNeg")
  false_positives = tf.reduce_sum(false_positive, name="FalsPos")
  false_negatives = tf.reduce_sum(false_negative, name="FalsNeg")

  loss = tf.reshape((false_positives * false_negatives - true_positives * true_negatives), [-1], name="Loss")
  real_loss = tf.reshape((false_positives * false_negatives - true_positives * true_negatives) / tf.sqrt(1 + (true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives)), [-1], name="RealLoss")

  optimizer = tf.train.AdamOptimizer().minimize(loss, name="Opt")

  clean_logits = tf.reshape(construct_network(False), [-1], name="Pred")
  tf.train.export_meta_graph(filename=(args.filename + ".meta"))

with tf.Session(graph=graph) as session:
  saver = tf.train.Saver(tf.all_variables())
  tf.initialize_all_variables().run()
  print('Initialized')
  saver.save(session, args.filename)
