import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import tensorflow as tf

NUMLABELS = 10
labrange = "ABCDEFGHIJ"

def show(x):
  plt.imshow(x)
  plt.set_cmap('gray')
  plt.show()

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=0)

# Transforms labels of form (X) with values 0 to NUM_LABELS - 1 into one-hot
# vectors.
def OneHot(labels):
  return (np. arange(NUMLABELS) == labels[:,None]).astype(np.float32)

# Transforms data in (X, 28, 28) to (X, 28 * 28).
def Flatten(dataset):
  return dataset.reshape((-1, 28 * 28)).astype(np.float32)

# Transforms data in (X, 28, 28) to (X, 28, 28, 1).
def MakeOneChannel(dataset):
  return dataset.reshape((-1, 28, 28, 1)).astype(np.float32)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def MakeStripeArrays(lgt, bias):
  dataset = np.random.rand(lgt, 28, 28)
  biasarr = np.zeros((10, 28, 28))
  labelsarr = np.zeros(lgt, dtype=np.int)
  for x in range(28):
    for y in range(28):
      biasarr[(y * 10) / 28][x][y] = 1
  for x in range(lgt):
    labelsarr[x] = x % 10
    dataset[x] += biasarr[x % 10]
  return dataset, labelsarr

def TransformOneDataset(dataset, labels, config):
  if config[0] == "Shaped":
    pass
  elif config[0] == "Flat":
    dataset = Flatten(dataset)
  elif config[0] == "Channel":
    dataset = MakeOneChannel(dataset)
  else:
    print("Invalid shape config value", config[0])
    sys.exit(1)
  if config[1] == "Labels":
    pass
  elif config[1] == "OneHot":
    labels = OneHot(labels)
  else:
    print("Invalid label config value", config[1])
    sys.exit(1)
  return dataset, labels

def TransformData(a, b, c, d, e, f, config):
  a, b = TransformOneDataset(a, b, config)
  c, d = TransformOneDataset(c, d, config)
  e, f = TransformOneDataset(e, f, config)
  return a, b, c, d, e, f

# Returns data in the shape of (X, 28, 28) and labels in (X) with values from
# 0 to NUM_LABELS - 1.
def LoadData(config):
  with open('notMNIST.pickle', 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save
    return TransformData(train_dataset, train_labels,
            valid_dataset, valid_labels,
            test_dataset, test_labels, config)

# Returns data in the shape of (X, 28, 28) and labels in (X) with values from
# 0 to NUM_LABELS - 1.
def LoadStripeData(bias, config):
  trd, trl = MakeStripeArrays(200000, bias)
  vld, vll = MakeStripeArrays(10000, bias)
  tsd, tsl = MakeStripeArrays(10000, bias)
  return TransformData(trd, trl, vld, vll, tsd, tsl, config)

def SaveConfig(filename, config):
  with open(filename + ".config", "wb") as f:
    pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)

def LoadGraph(filename):
  session = tf.Session()
  new_saver = tf.train.import_meta_graph(filename + ".meta")
  new_saver.restore(session, filename)
  with open(filename + ".config", "rb") as f:
    config = pickle.load(f)
  return session, new_saver, config
