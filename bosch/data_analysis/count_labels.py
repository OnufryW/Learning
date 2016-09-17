import sys
import numpy as np
import tensorflow as tf
from six.moves import range
from six.moves import cPickle as pickle

input_num = sys.argv[1] if len(sys.argv) > 1 else "0"

def LoadData(savefile):
  with open(savefile, "rb") as f:
    return pickle.load(f)

_, train_data, train_labels = LoadData("../data/train_numeric_%s.pickle" % (input_num,))

res = {}

for x in range(train_labels.shape[0]):
  if train_labels[x] in res:
    res[train_labels[x]] += 1
  else:
    res[train_labels[x]] = 1

print res
