from __future__ import print_function
import random
import numpy as np
import tensorflow as tf
from six.moves import range
import matplotlib.pyplot as plt
import tools

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = tools.LoadData()

train_dataset, train_labels = tools.Flatten(train_dataset, train_labels)
valid_dataset, valid_labels = tools.Flatten(valid_dataset, valid_labels)
test_dataset, test_labels = tools.Flatten(test_dataset, test_labels)

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.

train_subset = 10000

graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
  tf_train_labels = tf.constant(train_labels[:train_subset])
          
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = tf.Variable(
    tf.truncated_normal([28 * 28, tools.NUMLABELS]))
  biases = tf.Variable(tf.zeros([tools.NUMLABELS]))
  
  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 10001

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % tools.accuracy(
        predictions, train_labels[:train_subset, :]))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      print('Validation accuracy: %.1f%%' % tools.accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % tools.accuracy(test_prediction.eval(),
      test_labels))

  for x in range(10):
    resses = []
    for y in range(10):
      u = train_dataset[y,:]
      resses.append(weights.eval()[:,x].dot(train_dataset[y,:]))
    print(resses)
    z = weights.eval()[:,x]
    z.shape = (28, 28)
    print()
    print()
    print("Showing: ", tools.labrange[x])
    tools.show(z)

