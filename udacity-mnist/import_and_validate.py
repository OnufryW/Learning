from __future__ import print_function
import random
import numpy as np
import tensorflow as tf
from six.moves import range
import matplotlib.pyplot as plt
import tools
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else "DefaultGraph"
mode = sys.argv[2].lower() if len(sys.argv) > 2 else "valid"

session, _, config = tools.LoadGraph(filename)
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = tools.LoadData(config)

if mode == "valid":
  dataset = valid_dataset
  labels = valid_labels
elif mode == "test":
  dataset = test_dataset
  labels = test_labels
elif mode == "train":
  dataset = train_dataset
  labels = train_labels
else:
  print("Invalid mode: " + mode)
  print("Choices are: valid, test, train (default is valid)")
  sys.exit(1)

with session.as_default():
    prediction = tf.get_default_graph().get_tensor_by_name("Pred:0")
    input_tensor = tf.get_default_graph().get_tensor_by_name("Input:0")
    print('Accuracy in mode %s: %.1f%%' % (mode,
        tools.accuracy(prediction.eval(feed_dict={input_tensor : dataset}), labels)))
session.close()
