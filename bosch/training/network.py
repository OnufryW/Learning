from six.moves import range
import numpy as np
import tensorflow as tf
import sys
import random
import argparse

parser = argparse.ArgumentParser(description='Create and init the graph.')
parser.add_argument('-d', '--depth', default=2, type=int,
        help='Number of network layers')
parser.add_argument('--fraction_loss', dest='fraction_loss',
        action='store_true', help='Use Matthews coefficient as loss.')
parser.add_argument('--naive_loss', dest='fraction_loss',
        action='store_false', help='Use nominator of MCC as loss.')
parser.set_defaults(fraction_loss=False)
parser.add_argument('-k', '--keep_rate', default=0.9, type=float,
        help='Keep ratio of dropout layers')
parser.add_argument('--l2', default=0, type=float,
        help='The weight of l2 regularization we apply')
parser.add_argument('-r', '--ratio_stabilizer', default=0., type=float,
        help='Strength of the ratio stabilizer')
parser.add_argument('-w', '--width', default=1024, type=int,
        help='Width of a single layer')
parser.add_argument('filename', help='Filename for the graph')

args = parser.parse_args()

input_size=1936 
sdev=0.01

assert args.depth > 1
assert args.keep_rate > 0
assert args.keep_rate <= 1

with open(args.filename + ".log", "w") as logfile:
    logfile.write((
        "%s keep_rate: %f, depth: %d, width: %d, fraction_loss: %s, " + 
        "ratio_stabilizer: %f\n") % (
                args.filename, args.keep_rate, args.depth, args.width,
                args.fraction_loss, args.ratio_stabilizer))

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

  l2_reg = (tf.nn.l2_loss(weights[args.depth - 1]) +
          tf.nn.l2_loss(biases[args.depth - 1]))
  for x in range(args.depth - 1):
    l2_reg = l2_reg + tf.nn.l2_loss(weights[x])
    l2_reg = l2_reg + tf.nn.l2_loss(biases[x])
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

  logits = tf.reshape(tf.tanh(construct_network(True)) / 2. + 0.5,
          [-1], name="Logits")
  tf_train_labels = tf.placeholder(tf.float32, name="TrainLabels")

  true_positive = tf_train_labels * logits
  true_negative = (1. - tf_train_labels) * (1. - logits)
  false_positive = (1. - tf_train_labels) * logits
  false_negative = tf_train_labels * (1. - logits)

  true_positives = tf.reduce_sum(true_positive, name="TruePos")
  true_negatives = tf.reduce_sum(true_negative, name="TrueNeg")
  false_positives = tf.reduce_sum(false_positive, name="FalsPos")
  false_negatives = tf.reduce_sum(false_negative, name="FalsNeg")
  guess_true = tf.reduce_sum(logits, name="GuessTrue")
  guess_false = tf.reduce_sum(1. - logits, name="GuessFalse")
  total = guess_true + guess_false

  loss_entry = (false_positives * false_negatives -
          true_positives * true_negatives)
  if args.fraction_loss:
    loss_entry = loss_entry * loss_entry / (guess_true * guess_false + 0.1)
  norm_ratio = tf.reshape((guess_false + 0.001 * guess_true) /
          ((guess_true+ 0.001 * guess_false) * 172.),
          [-1], name="NormRatio")
  if args.ratio_stabilizer > 0:
    # That's 172 * predicted true / predicted false. We want that to be ~1.
    # The added term minimizes at norm_ratio = 1.
    loss_entry = loss_entry + args.ratio_stabilizer * (
        norm_ratio + (1. / norm_ratio) - 2)

  if args.l2 > 0:
    loss_entry = loss_entry + args.l2 * l2_reg

  loss = tf.reshape(loss_entry, [-1], name="Loss")

  real_loss = tf.reshape(
          (false_positives * false_negatives -
              true_positives * true_negatives) /
          tf.sqrt((guess_true) *
                        (true_positives + false_negatives) *
                        (true_negatives + false_positives) *
                        (guess_false)),
          [-1], name="RealLoss")

  optimizer = tf.train.AdamOptimizer().minimize(loss, name="Opt")

  clean_logits = tf.reshape(construct_network(False), [-1], name="Pred")
  tf.train.export_meta_graph(filename=(args.filename + ".meta"))

with tf.Session(graph=graph) as session:
  saver = tf.train.Saver(tf.all_variables())
  tf.initialize_all_variables().run()
  print('Initialized')
  saver.save(session, args.filename)
