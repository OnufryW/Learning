import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Display a graph of the loss')
parser.add_argument('graphname', help='Name of the graph')
parser.add_argument('--include_first_line', dest='include_first_line',
        action='store_true', help='Include first line in plot')
parser.add_argument('--noinclude_first_line', dest='include_first_line',
        action='store_false', help='Do not include first line in plot')
parser.add_argument('--plot_real_loss', dest='plot_real_loss',
        action='store_true', help='Plot the real loss')
parser.add_argument('--plot_computed_loss', dest='plot_real_loss',
        action='store_false', help='Plot the computed loss')
parser.set_defaults(include_first_line=False, plot_real_loss=True)
args = parser.parse_args()

test = []
valid = []
steps = []
read_line = args.include_first_line
with open(args.graphname + '.log', 'rt') as logfile:
  for line in logfile.readlines():
    if not read_line:
      read_line = True
      continue
    split_line = line.split(' ')
    if args.plot_real_loss:
      test.append(-float(split_line[3]))
      valid.append(-float(split_line[4]))
    else:
      test.append(-float(split_line[1]))
      valid.append(-float(split_line[2]))
    steps.append(int(split_line[0]))

plt.plot(steps, test, "r", steps, valid, "g")
plt.savefig(args.graphname + '.png')
plt.show()
