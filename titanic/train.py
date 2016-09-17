import sys
import numpy as np
import tensorflow as tf
from six.moves import range
from six.moves import cPickle as pickle

if len(sys.argv) < 3:
  print("Usage: %s FILENAME NUM_STEPS [BATCH_SIZE]" % sys.argv[0])
  sys.exit(1)

filename = sys.argv[1]
num_steps = int(sys.argv[2])
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 64

session = tf.Session()
saver = tf.train.import_meta_graph(filename + ".meta")
saver.restore(session, filename)

def LoadData(savefile):
  with open(savefile, "rb") as f:
    return pickle.load(f)

def Accuracy(predictions, labels):
  predval = predictions.ravel() > 0
  return (100. * np.sum(predval == (labels == 1)) / predictions.shape[0])

with open(filename + ".config", "r") as configfile:
  seed = int(configfile.read().strip())
  np.random.seed(seed=seed)

full_data, full_labels = LoadData("train.pickle")
print full_data.shape, full_labels.shape
test_data, test_labels = LoadData("test.pickle")
permutation = np.random.permutation(full_data.shape[0])
train_data = full_data[permutation[:650], :]
train_labels = full_labels[permutation[:650]]
valid_data = full_data[permutation[650:], :]
valid_labels = full_labels[permutation[650:]]

with session.as_default():
  with open(filename + ".log", "a") as logfile:
    graph = tf.get_default_graph()
    optimizer = graph.get_operation_by_name("Opt")
    loss = graph.get_tensor_by_name("Loss:0")
    pred = graph.get_tensor_by_name("Pred:0")
    inputt = graph.get_tensor_by_name("Input:0")
    labels = graph.get_tensor_by_name("TrainLabels:0")

    print "Beginning loss", loss.eval(feed_dict={inputt : train_data, labels : train_labels})

    for step in range(num_steps * 100):
      feed_dict = { inputt : train_data, labels : train_labels }
      session.run([optimizer, loss], feed_dict=feed_dict)

      if step % 100 == 0:
        train_loss, train_pred = session.run([loss, pred],
                feed_dict={inputt : train_data, labels: train_labels})
        train_acc = Accuracy(train_pred, train_labels)
        valid_loss, valid_pred = session.run([loss, pred],
                feed_dict={inputt : valid_data, labels: valid_labels})
        valid_acc = Accuracy(valid_pred, valid_labels)
        full_loss, full_pred = session.run([loss, pred],
                feed_dict={inputt : full_data, labels : full_labels})
        full_acc = Accuracy(full_pred, full_labels)
        logfile.write("%f %f %f %f %f %f\n" % (
            train_loss, valid_loss, full_loss, train_acc, valid_acc, full_acc))
        print('Loss --- train %.3f;    valid %.3f' % (train_loss, valid_loss))
        print('Accuracy --- train %.1f%%;    valid %.1f%%' % (
            train_acc, valid_acc))
  saver.save(session, filename)
session.close()

