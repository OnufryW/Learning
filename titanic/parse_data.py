import sys
import numpy as np
from six.moves import cPickle as pickle

def SplitLine(line):
  # Get rid of the name!
  li = line.strip().split("\"")
  return (li[0] + " " + li[-1]).split(",")

def ParseLine(line, labeled):
  li = SplitLine(line)
  if not labeled:
    li = [li[0], "-1"] + li[1:] 
  result = []
  # 0. We ignore the passenger number
  # 1. We copy the "survived" value
  result.append(int(li[1]))
  # 2. Copy the passenger class
  result.append(int(li[2]))
  # 3. We ignore the passenger's name.
  # 4. We put the sex as one variable.
  assert li[4] == "female" or li[4] == "male", "%s %s" % (str(li), str(labeled))
  result.append(int(li[4] == "female"))
  # 5. We just copy the age. Unknown goes to -20 (the idea being that if we
  #    cluster it far, it'll be easier for the network to notice).
  try:
    age = float(li[5])
  except Exception as e:
    age = -20.
  result.append(age)
  # 6. Grab the number of siblings / spouses
  result.append(int(li[6]))
  # 7. Grab the number of parents / children
  result.append(int(li[7]))
  # 8. Ignore the ticket number.
  # 9. Grab the fare.
  try:
    result.append(float(li[9]))
  except Exception as e:
    print li, e
    result.append(0.)
  # 10. Ignore the cabin number.
  # 11. One-hot the port of embarkation.
  for port in "CQS":
    result.append(int(li[11] == port))
  return result

def ParseFile(filename, labeled):
  with open(filename, "r") as f:
    data = []
    labels = []
    header = True
    for line in f.readlines():
      if header:
        header = False
        continue
      data.append(ParseLine(line, labeled)[1:])
      labels.append(ParseLine(line, labeled)[0])
  return np.array(data), np.array(labels)


train = ParseFile("train.csv", True)
train_mean = np.mean(train[0], axis=0)
train_std = np.var(train[0], axis=0)
normalized_train = (train[0] - train_mean) / train_std
with open("train.pickle", "wb") as f:
  pickle.dump((normalized_train, train[1]), f, pickle.HIGHEST_PROTOCOL)

test = ParseFile("test.csv", False)
normalized_test = (test[0] - train_mean) / train_std
with open("test.pickle", "wb") as f:
  pickle.dump((normalized_test, test[1]), f, pickle.HIGHEST_PROTOCOL)
