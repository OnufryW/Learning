import sys
import numpy as np
import tensorflow as tf
from six.moves import range
from six.moves import cPickle as pickle
import argparse
import datetime
import os

parser = argparse.ArgumentParser(description='Train the graph')
parser.add_argument('filename', help='Name of the graph to train')
parser.add_argument('threshold', default=0.5,
        help='Threshold to count a vote as True')
parser.add_argument('-i', '--input', default="1",
        help='Index of input data file')
args = parser.parse_args()

session = tf.Session()
saver = tf.train.import_meta_graph(args.filename + ".meta")
saver.restore(session, args.filename)

def LoadData(savefile):
  with open(savefile, "rb") as f:
    res = pickle.load(f)
    return res[1], res[2]

print 'Loading test data'
test_data, test_labels = LoadData("../data/train_numeric_%s.pickle" % (
    args.input,))
print 'Loaded training data'

with session.as_default():
  with open(args.filename + ".res.csv", "a") as resfile:
    graph = tf.get_default_graph()
    pred = graph.get_tensor_by_name("Pred:0")
    feed_dict = { inputt : test_data }
    test_pred = session.run([pred], feed_dict=feed_dict)
    test_val = np.tanh(test_pred) / 2. + 0.5
    test_choice = (test_val > args.threshold)
    pos = int(np.sum(test_choice))
    neg = int(np.sum(1 - test_choice))
    print 'Got %d YES and %d NO answers' % (pos, neg)
    for x in range(test_choice.shape[0]):
      resfile.write('%d,%d\n' % (test_labels[x], 1 if test_choice[x] else 0))

