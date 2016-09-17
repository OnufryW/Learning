from __future__ import print_function
import random
import numpy as np
import tensorflow as tf
from six.moves import range
import matplotlib.pyplot as plt
import tools
import sys

if len(sys.argv) < 3:
  print("Usage: %s FILENAME NUM_STEPS [BATCH_SIZE]" % sys.argv[0])
  sys.exit(1)

filename = sys.argv[1]
# TODO(onufry): Length should be expressed in minutes, not in steps.
num_steps = int(sys.argv[2])
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 128

session, saver, config = tools.LoadGraph(filename)

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = tools.LoadData(config)

with session.as_default():
  graph = tf.get_default_graph()
  optimizer = graph.get_operation_by_name("Opt")
  loss = graph.get_tensor_by_name("Loss:0")
  pred = graph.get_tensor_by_name("Pred:0")
  inputt = graph.get_tensor_by_name("Input:0")
  trainlabels = graph.get_tensor_by_name("TrainLabels:0")

  for step in range(num_steps):
    # TODO(onufry): Randomize better here.
    offset = (step * batch_size % (train_labels.shape[0] - batch_size))
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    feed_dict = { inputt : batch_data, trainlabels : batch_labels }
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)

    # TODO(onufry): This should be time-driven, maybe, or just out of here.
    if step % 500 == 0:
      print('Loss at step %d: %f' % (step, l))
      print('Train accuracy: %.1f%%' % tools.accuracy(pred.eval(
          feed_dict={inputt : batch_data }), batch_labels))
      print('Validation accuracy: %.1f%%' % tools.accuracy(pred.eval(
          feed_dict={inputt : valid_dataset }), valid_labels))
  print('Test accuracy: %.1f%%' % tools.accuracy(pred.eval(
      feed_dict={inputt : test_dataset}), test_labels))
  saver.save(session, filename)
session.close()
