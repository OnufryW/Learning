from __future__ import print_function
import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
import tools

TRAINSIZE = 20000
TESTSIZE = 10000

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = tools.LoadData()

models = []

for label in range(tools.NUMLABELS):
  clf = linear_model.LinearRegression()
  clf.fit(train_dataset[:TRAINSIZE].reshape(TRAINSIZE, 28*28),
          train_labels[:TRAINSIZE] == label)
  models.append(clf.coef_)

def Opinion(x):
  res = []
  for label in range(tools.NUMLABELS):
    res.append(models[label].ravel().dot(x))
  return tools.softmax(res)

def Choose(x):
  opin = Opinion(x.ravel())
  maxind = 0
  for x in range(1, tools.NUMLABELS):
    if opin[x] > opin[maxind]:
      maxind = x
  return tools.labrange[maxind]

def Accuracy():
  hit = 0.
  for x in range(TESTSIZE):
    if tools.labrange[test_labels[x]] == Choose(test_dataset[x]):
      hit += 1.
  print("Accuracy: {0:.0f}%".format(hit * 100. / TESTSIZE))


for x in range(tools.NUMLABELS):
  z = models[x]
  z.shape = (28, 28)
  print()
  print()
  print("Showing: ", tools.labrange[x])
  tools.show(z)

Accuracy()
