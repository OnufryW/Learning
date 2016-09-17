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

batch_size = 128
hidden_layers = 1024

graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 28 * 28))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, tools.NUMLABELS))
          
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights_1 = tf.Variable(tf.truncated_normal([28 * 28, hidden_layers]))
  weights_2 = tf.Variable(tf.truncated_normal([hidden_layers, tools.NUMLABELS]))
  biases = tf.Variable(tf.zeros([tools.NUMLABELS]))
  
  # Training computation.
  first_layer = tf.matmul(tf_train_dataset, weights_1)
  after_relu = tf.nn.relu(first_layer)
  logits = tf.matmul(after_relu, weights_2) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(tf.matmul(
      tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1)), weights_2) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(
      tf.nn.relu(tf.matmul(tf_test_dataset, weights_1)), weights_2) + biases)

num_steps = 10001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction],
            feed_dict=feed_dict)
    if (step % 500 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % tools.accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % tools.accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % tools.accuracy(test_prediction.eval(),
      test_labels))

  for x in range(10):
    z = weights_1.eval()[:,x]
    z.shape = (28, 28)
    print()
    print()
    print("Showing entry layer ", x)
    tools.show(z)

