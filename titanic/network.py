from six.moves import range
import numpy as np
import tensorflow as tf
import sys
import random

keep_rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
width = int(sys.argv[3]) if len(sys.argv) > 3 else 128
filename = sys.argv[4] if len(sys.argv) > 4 else "DefaultGraph"

input_size = 9
sdev=0.01

assert depth > 1
assert keep_rate > 0
assert keep_rate <= 1

with open(filename + ".log", "w") as logfile:
  logfile.write("%s %f %d %d\n" % (filename, keep_rate, depth, width))

seed = random.randint(0, 2 ** 30)
with open(filename + ".config", "w") as configfile:
  configfile.write(str(seed) + "\n")

graph = tf.Graph()
with graph.as_default():
  # Input data.
  input_dataset = tf.placeholder(tf.float32, name="Input")

  # Variables.
  weights = []
  biases = []

  weights.append(tf.Variable(
      tf.truncated_normal([input_size, width], stddev=sdev), name="weights1"))
  biases.append(tf.Variable(
      tf.truncated_normal([width], stddev=sdev), name="biases1"))
  for x in range(depth - 2):
    weights.append(tf.Variable(tf.truncated_normal(
        [width, width], stddev=sdev), name="weights"+str(x+2)))
    biases.append(tf.Variable(tf.truncated_normal(
        [width], stddev=sdev), name="biases"+str(x+2)))
  weights.append(tf.Variable(
      tf.truncated_normal([width, 1], stddev=sdev), name="weights"+str(depth)))
  biases.append(tf.Variable(
      tf.truncated_normal([1], stddev=sdev), name="biases"+str(depth)))

  # Let's do dropout on training to prevent overfitting.
  def construct_network(dropout):
    suffix = "D" if dropout else ""
    state = input_dataset
    for x in range(depth-1):
      operated = tf.matmul(
              state, weights[x], name="base"+str(x)+suffix) + biases[x]
      neuron = tf.nn.relu(operated, name="neuron"+str(x)+suffix)
      if dropout:
        state = tf.nn.dropout(neuron, keep_rate)
      else:
        state = neuron
    return tf.matmul(state, weights[-1], name="final"+suffix) + biases[-1]

  logits = tf.reshape(tf.tanh(construct_network(True)), [-1], name="Logits")
  tf_train_labels = tf.placeholder(tf.float32, name="TrainLabels")
  finvals = tf.reshape(
          (2 * tf_train_labels - 1) * logits / 2.01 + 0.5, [-1],
          name="FinLogits")
  loss_entry = tf.log(finvals, name="lentry")
  # It's not obvious how to define a good loss function :( What I want is
  # something that's close to zero below 0.5, and close to 1 over 0.5.
  loss = tf.reduce_mean(-loss_entry, name="Loss")
  optimizer = tf.train.AdamOptimizer().minimize(loss, name="Opt")

  clean_logits = tf.reshape(construct_network(False), [-1], name="Pred")
  tf.train.export_meta_graph(filename=(filename + ".meta"))

with tf.Session(graph=graph) as session:
  saver = tf.train.Saver(tf.all_variables())
  tf.initialize_all_variables().run()
  print('Initialized')
  saver.save(session, filename)
