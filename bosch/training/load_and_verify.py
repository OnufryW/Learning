import sys
import numpy as np
import tensorflow as tf
from six.moves import range
from six.moves import cPickle as pickle

session = tf.Session()
saver = tf.train.import_meta_graph("DefaultGraph.meta")
saver.restore(session, "DefaultGraph")

with session.as_default():
  print tf.get_default_graph().get_tensor_by_name("weights1:0").eval()[0]
