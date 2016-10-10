import sys
import numpy as np
import math
from six.moves import cPickle as pickle
import csv
import argparse

parser = argparse.ArgumentParser(
    description='Convert the .csv into a set of numpy table pickles')
parser.add_argument('--train', dest='mode', action='store_const', const='TRAIN',
        help='Parse train_numeric.csv')
parser.add_argument('--sample', dest='mode', action='store_const',
        const='SAMPLE', help='Parse part of train_numeric.csv into a sample')
parser.add_argument('--test', dest='mode', action='store_const', const='TEST',
        help='Pare test_numeric.csv')
        
args = parser.parse_args()

TRAIN_FILE = '../data/train_numeric.csv'
DATA_FILE = '../data/test_numeric.csv' if args.mode == 'TEST' else TRAIN_FILE
BLOB_SIZE = 5000 if args.mode == 'SAMPLE' else 50000
OUTPUT_FORMAT = "../data/%s_numeric_%%d.pickle" % (
        'test' if args.mode == 'TEST' else 'train')
miss_cols = set()

# Step one - calculate the averages and variances, for normalization
with open(TRAIN_FILE, "rt") as csvfile:
    reader = csv.reader(csvfile)
    rownum = 0
    for row in reader:
        if not rownum:
            header_row = row
            cols = len(header_row)
            counts = [0.] * cols
            sums = [0.] * cols
            squareSums = [0.] * cols
            avgs = [0.] * cols
            stddevs = [0.] * cols
            yes = [0.] * cols
            no = [0.] * cols
        else:
            for x in range(cols):
                if row[x]:
                    counts[x] += 1
                    sums[x] += float(row[x])
                    squareSums[x] += float(row[x]) * float(row[x])
        rownum += 1
    if args.mode != 'TEST':
      miss_cols.add(cols - 1)  # Skip the "response" row.
    miss_cols.add(0)  # Skip the ID row.
    for x in range(cols):
        if counts[x]:
            avgs[x] = sums[x] / counts[x]
            stddevs[x] = math.sqrt(squareSums[x] / counts[x] - avgs[x] * avgs[x])
            alpha = float(counts[x]) / float(rownum - 1)
            if alpha > 0:
                yes[x] = math.sqrt((1 - alpha) / alpha)
            if alpha < 1.:
                no[x] = math.sqrt(alpha / (1 - alpha))
            # Skip rows with almost no variance.
            if stddevs[x] < 0.000001:
                miss_cols.add(x)
        else:
            miss_cols.add(x)

print "Skipping %d columns, leaving %d, got %d rows in total." % (len(miss_cols), cols - len(miss_cols) - 1, rownum)

# Step two - parse the data, in ranges.
rowcount = rownum - 1
if args.mode == 'TEST':
  cols -= 1
with open(DATA_FILE, "rt") as csvfile:
    reader = csv.reader(csvfile)
    rownum = 0
    outrownum = 0
    outcols = 2 * (cols - len(miss_cols))
    data_arr = np.empty((BLOB_SIZE, outcols))
    data_blob_count = 0
    labels_arr = np.empty(BLOB_SIZE)
    for row in reader:
        if rownum:
            colnum = 0
            for x in range(cols):
                if x not in miss_cols:
                    if row[x]:
                        data_arr[outrownum][colnum] = (float(row[x]) - avgs[x]) / stddevs[x]
                        data_arr[outrownum][colnum + 1] = yes[x]
                    else:
                        data_arr[outrownum][colnum] = 0.
                        data_arr[outrownum][colnum + 1] = no[x]
                    colnum += 2
            if args.mode == 'TEST':
              labels_arr[outrownum] = float(row[0])
            else:
              labels_arr[outrownum] = float(row[cols-1])
            outrownum += 1
        rownum += 1
        if outrownum == data_arr.shape[0]:
            print "Pickling blob %d, with shape " % data_blob_count, data_arr.shape
            print "Before anything, we have %d rows left" % (rowcount,)
            outrownum = 0
            rowcount -= data_arr.shape[0]
            # Pickle the data.
            filename = OUTPUT_FORMAT % data_blob_count
            if args.mode == 'SAMPLE':
                filename = "../data/train_numeric_99.pickle"
            with open(filename, "wb") as f:
                pickle.dump((header_row, data_arr, labels_arr), f)
            data_blob_count += 1
            data_arr = np.empty((min(BLOB_SIZE, rowcount), outcols))
            labels_arr = np.empty(min(BLOB_SIZE, rowcount))
            if args.mode == 'SAMPLE':
                sys.exit(0)
