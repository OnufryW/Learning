import sys
import numpy as np
import tensorflow as tf
from six.moves import range
from six.moves import cPickle as pickle
import argparse
import datetime
import os

parser = argparse.ArgumentParser(description='Train the graph')
parser.add_argument('filename', help='Name of the graph to train')
parser.add_argument('steps_or_seconds', type=int,
        help='Number of steps/seconds to take (controlled by -s)')
parser.add_argument('-b', '--batch_size', default=6000, type=int,
        help='Size of single training step batch')
parser.add_argument('-i', '--input', default="1",
        help='Index of input data file')
parser.add_argument('--log_every_seconds', default=5, type=int,
        help='Every how many seconds we log? Ignored if seconds=False.')
parser.add_argument('--log_every_steps', default=10, type=int,
        help='Every how many steps we log? Ignored if seconds=True.')
parser.add_argument('--seconds', dest='seconds', action='store_true',
        help='Use clock time to determine when to log and terminate')
parser.add_argument('--steps', dest='seconds', action='store_false',
        help='Use step count to determine when to log and terminate')
parser.add_argument('-v', '--validation', default="0",
        help='Index of validation data file')
parser.set_defaults(seconds=False)

args = parser.parse_args()

session = tf.Session()
saver = tf.train.import_meta_graph(args.filename + ".meta")
saver.restore(session, args.filename)

def LoadData(savefile):
  with open(savefile, "rb") as f:
    res = pickle.load(f)
    return res[1], res[2]

with open(args.filename + ".config", "r") as configfile:
  seed = int(configfile.read().strip())
  np.random.seed(seed=seed)

if not os.path.isfile(args.filename + ".steps"):
  step = 0
else:
  with open(args.filename + ".steps", "r") as stepsfile:
    step = int(stepsfile.read().strip())

print 'Loading input data'
train_data, train_labels = LoadData("../data/train_numeric_%s.pickle" % (
    args.input,))
print 'Loaded training data'
valid_data, valid_labels = LoadData("../data/train_numeric_%s.pickle" % (
    args.validation,))
print 'Loaded validation data'
print 'Train data shape:', train_data.shape
print 'Train labels shape:', train_labels.shape

time_at_begin = datetime.datetime.now()
def NeedToEnd(steps):
  global time_at_begin
  if args.seconds:
    time_passed = datetime.datetime.now() - time_at_begin
    return time_passed > datetime.timedelta(seconds=args.steps_or_seconds)
  else:
    return steps > args.steps_or_seconds

time_at_last_log = datetime.datetime.now()
def NeedToLog(steps):
  global time_at_last_log
  if args.seconds:
    time_passed = datetime.datetime.now() - time_at_last_log
    if time_passed > datetime.timedelta(seconds=args.log_every_seconds):
      time_at_last_log = datetime.datetime.now()
      return True
    return False
  else:
    return steps % args.log_every_steps == 0

with session.as_default():
  with open(args.filename + ".log", "a") as logfile:
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
    norm_ratio = graph.get_tensor_by_name("NormRatio:0")
    guess_true = graph.get_tensor_by_name("GuessTrue:0")
    guess_false = graph.get_tensor_by_name("GuessFalse:0")

    print "Beginning loss", loss.eval(feed_dict={
        inputt : train_data, labels : train_labels})

    while not NeedToEnd(step):
      offset = step * args.batch_size % (
              train_labels.shape[0] - args.batch_size)
      batch_data = train_data[offset:(offset + args.batch_size), :]
      batch_labels = train_labels[offset:(offset + args.batch_size)]
      feed_dict = { inputt : batch_data, labels : batch_labels }
      _, batch_loss = session.run([optimizer, loss], feed_dict=feed_dict)
      print "Loss at step %d is %.5f" % (step, batch_loss)

      if NeedToLog(step):
        (t_loss, t_real_loss, t_true_pos, t_true_neg, t_false_pos,
                t_false_neg, t_norm_ratio, t_g_true, t_g_false) = session.run(
                        [loss, real_loss, true_pos, true_neg, false_pos,
                            false_neg, norm_ratio, guess_true, guess_false],
                        feed_dict={inputt : train_data, labels: train_labels})
        (v_loss, v_real_loss, v_true_pos, v_true_neg, v_false_pos, v_false_neg,
                v_norm_ratio, v_g_true, v_g_false) = session.run(
                        [loss, real_loss, true_pos, true_neg, false_pos,
                            false_neg, norm_ratio, guess_true, guess_false],
                        feed_dict={inputt : valid_data, labels: valid_labels})
        logfile.write("%d %f %f %f %f %f %f %f %f %f %f %f %f\n" % (
            step, t_loss, v_loss, t_real_loss, v_real_loss, t_true_pos,
            t_true_neg, t_false_pos, t_false_neg, v_true_pos, v_true_neg,
            v_false_pos, v_false_neg))
        print('Loss --- train %.5f;    valid %.5f' % (t_loss, v_loss))
        print('Real loss --- train %.5f;    valid %.5f' % (
            t_real_loss, v_real_loss))
        print(('True values hit --- train %.1f%% (%f / %f);    ' +
                'valid %.1f%% (%f / %f)') % (
                  100 * t_true_pos / (t_true_pos + t_false_neg),
                  t_true_pos, t_true_pos + t_false_neg,
                  100 * v_true_pos / (v_true_pos + v_false_neg), v_true_pos,
                  v_true_pos + v_false_neg))
        print(('False values hit --- train %.1f%% (%f / %f);   ' + 
                'valid %.1f%% (%f / %f)') % (
                    100 * t_true_neg / (t_true_neg + t_false_pos), t_true_neg,
                    t_true_neg + t_false_pos,
                    100 * v_true_neg / (v_true_neg + v_false_pos), v_true_neg,
                    v_true_neg + v_false_pos))
        print('Norm ratio (train): %f = 172 * %f / %f' % (
            t_norm_ratio, t_g_true, t_g_false))
        print('Norm ratio (valid): %f = 172 * %f / %f' % (
            v_norm_ratio, v_g_true, v_g_false))
      step += 1
  saver.save(session, args.filename)
session.close()

with open(args.filename + ".steps", "w") as stepsfile:
  stepsfile.write(str(step) + "\n")
