import sys
import numpy as np
import tensorflow as tf
from six.moves import range
from six.moves import cPickle as pickle
import argparse
import datetime
import math
import os

parser = argparse.ArgumentParser(description='Chooser of threshold')
parser.add_argument('filename', help='Name of the graph to use')
parser.add_argument('-v', '--validation', default="0",
        help='Index of validation data file')

args = parser.parse_args()

session = tf.Session()
saver = tf.train.import_meta_graph(args.filename + '.meta')
saver.restore(session, args.filename)

def LoadData(savefile):
  with open(savefile, 'rb') as f:
    return pickle.load(f)

print 'Loading validation data'
_, valid_data, valid_labels = LoadData('../data/train_numeric_%s.pickle' % (
    args.validation,))
print 'Loaded validation data'

def CorrRes(pred_val, real_val, thresh):
  pred_res = (pred_val > thresh)
  print np.sum(pred_res), np.sum(pred_val), pred_val[0][0], pred_val[0][1], pred_res[0][0], pred_res[0][1]
  true_pos = np.sum(pred_res * real_val)
  false_pos = np.sum(pred_res * (1 - real_val))
  true_neg = np.sum((1 - pred_res) * (1 - real_val))
  false_neg = np.sum((1 - pred_res) * real_val)
  print 'For threshold %f true_pos is %d, true_neg %d, false_pos %d, false_neg %d\n' % (thresh, true_pos, true_neg, false_pos, false_neg)
  if (true_pos + false_pos == 0) or (true_neg + false_neg == 0):
    return 0.
  return (true_pos * true_neg - false_pos * false_neg) / math.sqrt(
          (true_pos + false_pos) * (true_pos + false_neg) *
          (true_neg + false_pos) * (true_neg + false_neg))

def ChooseThreshold(pred_val, real_val, thresh_list):
  best_thresh = 0
  best_val = 0
  for thresh in thresh_list:
    new_val = CorrRes(pred_val, real_val, thresh)
    if new_val >= best_val:
      best_val = new_val
      best_thresh = thresh
  return (best_thresh, best_val)

def ChooseThresholdReal(pred_val, real_val):
  threshes = [0.01 * x for x in range(1, 100)]
  best_first, val = ChooseThreshold(pred_val, real_val, threshes)
  print 'Best suggestion is to use threshold %f, result %f' % (best_first, val)
  return best_first

with session.as_default():
  graph = tf.get_default_graph()
  pred = graph.get_tensor_by_name('Pred:0')
  inputt = graph.get_tensor_by_name('Input:0')
  my_pred = session.run([pred], feed_dict={inputt : valid_data})
  rpred = np.tanh(my_pred) / 2. + 0.5
  threshold = ChooseThresholdReal(rpred, valid_labels)
