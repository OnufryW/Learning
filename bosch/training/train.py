import sys
import numpy as np
import tensorflow as tf
from six.moves import range
from six.moves import cPickle as pickle

if len(sys.argv) < 3:
  print("Usage: %s FILENAME NUM_STEPS [INPUT_NUM] [VALIDATION_NUM] [BATCH_SIZE]" % sys.argv[0])
  sys.exit(1)

filename = sys.argv[1]
num_steps = int(sys.argv[2])
batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
input_num = sys.argv[3] if len(sys.argv) > 3 else "1"
valid_num = sys.argv[4] if len(sys.argv) > 4 else "0"

session = tf.Session()
saver = tf.train.import_meta_graph(filename + ".meta")
saver.restore(session, filename)

def LoadData(savefile):
  with open(savefile, "rb") as f:
    return pickle.load(f)

with open(filename + ".config", "r") as configfile:
  seed = int(configfile.read().strip())
  np.random.seed(seed=seed)

_, train_data, train_labels = LoadData("../data/train_numeric_%s.pickle" % (input_num,))
_, valid_data, valid_labels = LoadData("../data/train_numeric_%s.pickle" % (valid_num,))
_, test_data, test_labels = LoadData("../data/train_numeric_11.pickle")
print train_data.shape, train_labels.shape

with session.as_default():
  with open(filename + ".log", "a") as logfile:
    graph = tf.get_default_graph()
    optimizer = graph.get_operation_by_name("Opt")
    loss = graph.get_tensor_by_name("Loss:0")
    real_loss = graph.get_tensor_by_name("RealLoss:0")
    pred = graph.get_tensor_by_name("Pred:0")
    inputt = graph.get_tensor_by_name("Input:0")
    labels = graph.get_tensor_by_name("TrainLabels:0")
    true_pos = graph.get_tensor_by_name("TruePos:0")
    true_neg = graph.get_tensor_by_name("TrueNeg:0")
    false_pos = graph.get_tensor_by_name("FalsPos:0")
    false_neg = graph.get_tensor_by_name("FalsNeg:0")

    print "Beginning loss", loss.eval(feed_dict={inputt : train_data, labels : train_labels})

    for step in range(num_steps * 100):
      offset = step * batch_size % (train_labels.shape[0] - batch_size)
      batch_data = train_data[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size)]
      feed_dict = { inputt : batch_data, labels : batch_labels }
      _, looo = session.run([optimizer, loss], feed_dict=feed_dict)
      print "Loss at step %d is %.5f" % (step, looo)

      if step % 10 == 0:
        train_loss, train_real_loss, train_true_pos, train_true_neg, train_false_pos, train_false_neg = session.run([loss, real_loss, true_pos, true_neg, false_pos, false_neg],
                feed_dict={inputt : train_data, labels: train_labels})
        valid_loss, valid_real_loss, valid_true_pos, valid_true_neg, valid_false_pos, valid_false_neg = session.run([loss, real_loss, true_pos, true_neg, false_pos, false_neg],
                feed_dict={inputt : valid_data, labels: valid_labels})
        logfile.write("%f %f %f %f %f %f %f %f %f %f %f %f\n" % (
            train_loss, valid_loss, train_real_loss, valid_real_loss, train_true_pos, train_true_neg, train_false_pos, train_false_neg, valid_true_pos, valid_true_neg, valid_false_pos, valid_false_neg))
        print('Loss --- train %.5f;    valid %.5f' % (train_loss, valid_loss))
        print('Real loss --- train %.5f;    valid %.5f' % (train_real_loss, valid_real_loss))
        print('True values hit --- train %.1f%% (%f / %f);    valid %.1f%% (%f / %f)' % (
            100 * train_true_pos / (train_true_pos + train_false_neg), train_true_pos, train_true_pos + train_false_neg, 100 * valid_true_pos / (valid_true_pos + valid_false_neg), valid_true_pos, valid_true_pos + valid_false_neg))
        print('False values hit --- train %.1f%% (%f / %f);   valid %.1f%% (%f / %f)' % (
            100 * train_true_neg / (train_true_neg + train_false_pos), train_true_neg, train_true_neg + train_false_pos, 100 * valid_true_neg / (valid_true_neg + valid_false_pos), valid_true_neg, valid_true_neg + valid_false_pos))
  saver.save(session, filename)
session.close()

