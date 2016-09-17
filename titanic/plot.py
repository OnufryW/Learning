import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) not in (2, 3):
  print "Usage: %s GRAPHNAME [STEP_SIZE]" % sys.argv[0]
  sys.exit(1)

step_size = sys.argv[2] if len(sys.argv) > 2 else 100

losses = []
accuracies = []

with open(sys.argv[1] + ".log", "r") as logfile:
  first = True
  num = 0
  for line in logfile.readlines():
    if first:
      first = False
      continue
    split = line.strip().split(" ")
    losses.append([float(split[0]), float(split[1]), float(split[2])])
    accuracies.append([float(split[3]), float(split[4]), float(split[5])])
    num += step_size 

ls = np.array(losses)
ac = np.array(accuracies)

plt.plot(ls)
plt.show()
plt.plot(ac)
plt.show()
