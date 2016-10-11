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

def add_var_summaries(var, name):
  with tf.name_scope("summaries"):
    mean = tf.reduce_mean(var)
    tf.scalar_summary(name + "/mean", mean)
    with tf.name_scope("stddev"):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary(name + "/stddev", stddev)
    tf.scalar_summary(name + "/max", tf.reduce_max(var))
    tf.scalar_summary(name + "/min", tf.reduce_min(var))
    tf.histogram_summary(name, var)

def weight_relu(inp, w_shape, b_shape, keep_prob, layer_name):
  with tf.name_scope("weights"):
    weights = tf.Variable(tf.truncated_normal(w_shape, stddev=sdev),
            name="weights")
    add_var_summaries(weights, layer_name + "/weights")
  with tf.name_scope("biases"):
    biases = tf.Variable(tf.truncated_normal(b_shape, stddev=sdev),
            name="biases")
    add_var_summaries(biases, layer_name + "/biases")
  with tf.name_scope("l2_reg"):
    l2_reg = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
    add_var_summaries(l2_reg, layer_name + "/l2_reg")
  with tf.name_scope("operation"):
    op = tf.matmul(inp, weights, name="multiplied")
    tf.histogram_summary(layer_name + "/pre-activate", op)
    neuron = tf.nn.relu(op, name="neuron")
    tf.histogram_summary(layer_name + "/post-activate", neuron)
  dropped = tf.nn.dropout(neuron, keep_prob, name="dropped")
  tf.histogram_summary(layer_name + "/after-dropout", dropped)
  return (dropped, l2_reg)


graph = tf.Graph()
with graph.as_default():
  # Input data.
  input_dataset = tf.placeholder(tf.float32, name="Input")
  keep_prob = tf.placeholder(tf.float32, name="KeepProb")

  with tf.name_scope("BaseLayer"):
    (state, l2_reg) = weight_relu(input_dataset, [input_size, args.width],
        [args.width], keep_prob, "BaseLayer")
  for x in range(args.depth - 2):
    with tf.name_scope("HiddenLayer" + str(x + 1)):
      state, new_l2 = weight_relu(state, [args.width, args.width],
          [args.width], keep_prob, "HiddenLayer" + str(x + 1))
      l2_reg = l2_reg + new_l2
  state, new_l2 = weight_relu(state, [args.width, 1], [1], keep_prob,
      "FinalLayer")
  l2_reg = l2_reg + new_l2
  
  logits = tf.reshape(tf.tanh(state) / 2. + 0.5, [-1], name="Logits")
  add_var_summaries(logits, "Logits")

  tf_train_labels = tf.placeholder(tf.float32, name="TrainLabels")
  true_positive = tf_train_labels * logits
  add_var_summaries(true_positive, "TruePositives")
  true_negative = (1. - tf_train_labels) * (1. - logits)
  add_var_summaries(true_negative, "TrueNegative")
  false_positive = (1. - tf_train_labels) * logits
  add_var_summaries(false_positive, "FalsePositives")
  false_negative = tf_train_labels * (1. - logits)
  add_var_summaries(false_negative, "FalseNegative")

  true_positives = tf.reduce_sum(true_positive, name="TruePos")
  true_negatives = tf.reduce_sum(true_negative, name="TrueNeg")
  false_positives = tf.reduce_sum(false_positive, name="FalsPos")
  false_negatives = tf.reduce_sum(false_negative, name="FalsNeg")
  guess_true = tf.reduce_sum(logits, name="GuessTrue")
  guess_false = tf.reduce_sum(1. - logits, name="GuessFalse")
  total = guess_true + guess_false

  loss_entry = (false_positives * false_negatives -
          true_positives * true_negatives)
  tf.scalar_summary("LossNumerator", loss_entry)
  if args.fraction_loss:
    loss_entry = loss_entry * loss_entry / (guess_true * guess_false + 0.1)
  norm_ratio = tf.reshape((guess_false + 0.001 * guess_true) /
          ((guess_true+ 0.001 * guess_false) * 172.),
          [-1], name="NormRatio")
  tf.scalar_summary(["NormRatio"], norm_ratio)
  norm_error_term = norm_ratio + (1. / norm_ratio) - 2
  tf.scalar_summary(["NormRatioEntry"], norm_error_term)
  tf.scalar_summary(["ScaledNormRatioEntry"],
          args.ratio_stabilizer * norm_error_term)
  if args.ratio_stabilizer > 0:
    # That's 172 * predicted true / predicted false. We want that to be ~1.
    # The added term minimizes at norm_ratio = 1.
    loss_entry = loss_entry + args.ratio_stabilizer * norm_error_term

  tf.scalar_summary("TotalL2Entry", l2_reg)
  tf.scalar_summary("ScaledTotalL2Entry", args.l2 * l2_reg)
  if args.l2 > 0:
    loss_entry = loss_entry + args.l2 * l2_reg

  loss = tf.reshape(loss_entry, [-1], name="Loss")
  tf.scalar_summary(["TotalLoss"], loss)

  real_loss = tf.reshape(
          (false_positives * false_negatives -
              true_positives * true_negatives) /
          tf.sqrt((guess_true) *
                        (true_positives + false_negatives) *
                        (true_negatives + false_positives) *
                        (guess_false)),
          [1], name="RealLoss")
  print real_loss
  tf.scalar_summary(["RealLossValue"], real_loss)

  optimizer = tf.train.AdamOptimizer().minimize(loss, name="Opt")
  merged = tf.merge_all_summaries()
  print merged.name

  tf.train.export_meta_graph(filename=(args.filename + ".meta"))

with tf.Session(graph=graph) as session:
  saver = tf.train.Saver(tf.all_variables())
  tf.initialize_all_variables().run()
  print('Initialized')
  saver.save(session, args.filename)
