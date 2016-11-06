""" TF rewrite of reinforcement learning agent written by Andrei Karpathy. """

import numpy as np
import shutil
from six.moves import range
from six.moves import cPickle as pickle
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import os
import os.path
import atari_env
import fake_env

parser = argparse.ArgumentParser(description='Train the pong!')
parser.add_argument('--render', help='Render the game', action='store_true')
parser.add_argument('--width', help='Width of hidden layer', default=200,
        type=int)
parser.add_argument('--gamma', help='Reward discount factor over time',
        default=0.99, type=float)
parser.add_argument('--decay_rate', help='Decay rate of gradient memory',
        default='0.99', type=float)
parser.add_argument('--learning_rate', help='Learning rate', default=0.0001,
        type=float)
parser.add_argument('--logdir', help='Base directory for logs',
        default='/tmp/pong')
parser.add_argument('--learn_interval', default=10, type=int,
        help='Number of full games (of 21) between learning')
parser.add_argument('--stat_interval', default=10, type=int,
        help='Number of full games (of 21) between exporting stats to TB')
parser.add_argument('--save_interval', default=100, type=int,
        help='Number of full games (of 21) between checkpointing vars to disk')
parser.add_argument('--new_start', action='store_true', default=False,
        help='Delete the old checkpoint and start anew')
parser.add_argument('--checkpoint', default='pong_checkpoint',
        help='Checkpoint file path')
parser.add_argument('--log_device_placement', action='store_true',
        help='Inform on what device is the operation running')
args = parser.parse_args()

env = atari_env.AtariEnv()
# env = fake_env.FakeEnv(0.0, -2.)

def show(x):
  plt.imshow(x)
  plt.show()

if args.new_start:
  print 'Removing old checkpoint files'
  shutil.rmtree(args.logdir, ignore_errors=True)
  if os.path.exists(args.checkpoint):
    os.remove(args.checkpoint)
    os.remove(args.checkpoint + '.episode')

def add_var_summaries(var, name):
  with tf.name_scope("summaries"):
    mean = tf.reduce_mean(var)
    tf.scalar_summary(name + "/mean", mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)), name="sumstdv")
    tf.scalar_summary(name + "/stddev", stddev)
    tf.scalar_summary(name + "/max", tf.reduce_max(var))
    tf.scalar_summary(name + "/min", tf.reduce_min(var))
    tf.histogram_summary(name, var)

# TODO(onufry): Should there be an L2 reg in here somewhere? Let's look how
# it moves first.
def weight_relu(inp, w_shape, layer_name, should_relu):
  with tf.name_scope(layer_name):
    w_dev = 1. / np.sqrt(w_shape[0])
    weights = tf.Variable(tf.truncated_normal(w_shape, stddev=w_dev))
    add_var_summaries(weights, layer_name + "/weights")
    l2 = tf.nn.l2_loss(weights) / w_shape[1]  # Normalize so the loss is ~1.
    tf.scalar_summary(layer_name + "/l2_loss", l2)
#    biases = tf.Variable(tf.truncated_normal([w_shape[1]], stddev=0.1))
#    add_var_summaries(biases, layer_name + "/biases")
#    op = tf.add(tf.matmul(inp, weights), biases)
    op = tf.matmul(inp, weights)
    if should_relu:
      add_var_summaries(op, layer_name + "/pre-activate")
      neuron = tf.nn.relu(op)
    else:
      neuron = op
    add_var_summaries(neuron, layer_name + "/post-activate")
    return (neuron, l2)

def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, len(r))):
    if r[t]: running_add = 0 # Reset sum, since at game boundary.
    running_add = running_add * args.gamma + r[t]
    discounted_r[t] = running_add
  # Now, normalize.
  discounted_r -= np.mean(discounted_r)
  discounted_r /= np.std(discounted_r)
  return discounted_r

graph = tf.Graph()
with graph.as_default():
  input_image = tf.placeholder(tf.float32, name="Input")
  add_var_summaries(input_image, "Input")
  hidden, l2h = weight_relu(
          input_image, [env.D, args.width], "HiddenLayer", True)
  result, l2f = weight_relu(hidden, [args.width, 1], "FinalLayer", False)
  end_val = tf.sigmoid(result)
  add_var_summaries(end_val, "Probabilities")

  # We add the placeholders for the sole purpose of exporting them in TB.
  rewards_ph = tf.placeholder(tf.float32, name="Rewards")
  add_var_summaries(rewards_ph, "Rewards")
  discounted_ph = tf.placeholder(tf.float32, name="DiscountedRewards")
  add_var_summaries(discounted_ph, "DiscountedRewards")
  choices_ph = tf.placeholder(tf.float32, name="ChoicesMade")
  add_var_summaries(choices_ph, "Choices")

  # The real learning part.
  change_volume = tf.reshape(
          tf.log(0.001 + 0.999 * tf.abs(choices_ph - end_val)), [-1])
  loss_entries = change_volume * tf.reshape(discounted_ph, [-1])
  add_var_summaries(loss_entries, "Loss")
  loss = tf.reduce_sum(loss_entries)
  tf.scalar_summary("Loss/initial_value", loss)
  final_loss = loss + 50 * (l2h + l2f)
  tf.scalar_summary("Loss/final_value", final_loss)

  optimizer = tf.train.AdamOptimizer()
#  optimizer = tf.train.AdadeltaOptimizer(learning_rate=args.learning_rate,
#          rho=args.decay_rate)
#  optimizer = tf.train.GradientDescentOptimizer(
#          learning_rate=args.learning_rate)
  grads = optimizer.compute_gradients(loss)
  for grad in grads:
    add_var_summaries(grad[0], "GradientOf" + grad[1].name)
  minimizer = optimizer.apply_gradients(grads)
  merged = tf.merge_all_summaries()
  init = tf.initialize_all_variables()
  log_writer = tf.train.SummaryWriter(args.logdir, graph)
  saver = tf.train.Saver()

  # We store inputs, choices and rewards in two copies - one to be managed by
  # the stats exporting system, the other one for learning.
  inputs, choices, rewards = [[], []], [[], []], [[], []]
  # Now we train.
  observation = env.reset()
  running_reward = -env.game_length
  reward_sum = 0

  config = tf.ConfigProto(log_device_placement=args.log_device_placement)
  session = tf.Session(config=config)
  with session.as_default():
    # Initialize the variables.
    episode_number = 0
    if not args.new_start and os.path.exists(args.checkpoint):
      saver.restore(session, args.checkpoint)
      with open(args.checkpoint + '.episode', 'rt') as f:
        episode_number = int(f.read())
    else:
      session.run(init)

    while True:
      if args.render: env.render()

      inp = observation.ravel()

      up_prob = session.run([end_val],
                            feed_dict={input_image : inp.reshape((1,env.D))})[0]

      action = 2 if np.random.uniform() > up_prob else 3
      observation, reward, done = env.step(action)
      # Store for later reference:
      reward_sum += reward
      for ind in (0, 1):
        choices[ind].append(action - 2)
        inputs[ind].append(inp)
        rewards[ind].append(reward)

      if done:
        # Bookkeep
        running_reward = running_reward * 0.99 + reward_sum * 0.01
        print 'resetting env. episode reward total was %f. running mean %f' % (
                reward_sum, running_reward)
        reward_sum = 0
        observation = env.reset()
        episode_number += 1

        # Periodically export stats.
        if (episode_number + 1) % args.stat_interval == 0:
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          summaries, _, _ = session.run([merged, end_val, loss],
                  feed_dict = {
                      input_image : np.vstack(inputs[1]),
                      rewards_ph : np.vstack([x for x in rewards[1] if x]),
                      choices_ph : np.vstack(choices[1]),
                      discounted_ph : discount_rewards(rewards[1])},
                  options=run_options,
                  run_metadata=run_metadata)
          log_writer.add_run_metadata(run_metadata, 'Epis. %d' % episode_number)
          log_writer.add_summary(summaries, episode_number)
          inputs[1], rewards[1], choices[1] = [], [], []

        # Learn.
        if (episode_number + 1) % args.learn_interval == 0:
          feed_dict = {input_image: np.vstack(inputs[0]),
                       choices_ph : np.vstack(choices[0]),
                       discounted_ph : discount_rewards(rewards[0])}
          print len(inputs[0]), len(choices[0]), len(rewards[0])
          _ = session.run(minimizer, feed_dict=feed_dict)
          inputs[0], choices[0], rewards[0] = [], [], []

        # Checkpoint variables to disk.
        if (episode_number + 1) % args.save_interval == 0:
          saver.save(session, args.checkpoint)
          with open(args.checkpoint + '.episode', 'wt') as f:
            f.write(str(episode_number))

      if reward:
        print 'ep. %d game done, reward %f' % (episode_number, reward)
