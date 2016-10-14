""" TF rewrite of reinforcement learning agent written by Andrei Karpathy. """

import numpy as np
from six.moves import range
from six.moves import cPickle as pickle
import tensorflow as tf
import argparse
import gym
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Train the pong!')
parser.add_argument('--render', help='Render the game', action='store_true')
parser.add_argument('--width', help='Width of hidden layer', default=200,
        type=int)
parser.add_argument('--gamma', help='Reward discount factor over time',
        default=0.99, type=float)
parser.add_argument('--decay_rate', help='Decay rate of gradient memory',
        default='0.95', type=float)
parser.add_argument('--learning_rate', help='Learning rate', default=0.0001,
        type=float)
parser.add_argument('--logdir', help='Base directory for logs', default='/tmp')
parser.add_argument('--batch_size', help='Number of games between learning',
        default=10, type=int)
args = parser.parse_args()

env = gym.make("Pong-v0")

def show(x):
  plt.imshow(x)
  plt.show()

D = 80 * 80  # The size of the preprocessed image.
def prepro(I):
  """ Preprocess the image just as the original code does it. """
  I = I[35:195]
  I = I[::2,::2,0]
  I[I == 144] = 0
  I[I == 109] = 0
  I[I != 0] = 1
  # In the future, likely not ravel, but convo.
  return I.astype(np.float).ravel()

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
    biases = tf.Variable(tf.truncated_normal([w_shape[1]], stddev=0.1))
    add_var_summaries(biases, layer_name + "/biases")
    op = tf.add(tf.matmul(inp, weights), biases)
    if should_relu:
      add_var_summaries(op, layer_name + "/pre-activate")
      neuron = tf.nn.relu(op)
    else:
      neuron = op
    add_var_summaries(neuron, layer_name + "/post-activate")
    return neuron

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
  hidden = weight_relu(input_image, [D, args.width], "HiddenLayer", True)
  result = weight_relu(hidden, [args.width, 1], "FinalLayer", False)
  end_val = tf.sigmoid(result)
  add_var_summaries(end_val, "Probabilities")

  # We add the placeholders for the sole purpose of exporting them in TB.
  rewards_ph = tf.placeholder(tf.float32)
  add_var_summaries(rewards_ph, "Rewards")
  discounted_ph = tf.placeholder(tf.float32)
  add_var_summaries(discounted_ph, "DiscountedRewards")
  choices_ph = tf.placeholder(tf.float32)
  add_var_summaries(choices_ph, "Choices")

  # The real learning part.
  loss_entries = (choices_ph - end_val) * discounted_ph
  add_var_summaries(loss_entries, "Loss")
  loss = tf.reduce_sum(loss_entries)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
#  optimizer = tf.train.AdadeltaOptimizer(learning_rate=args.learning_rate,
#          rho=args.decay_rate)
  grads = optimizer.compute_gradients(loss)
  for grad in grads:
    add_var_summaries(grad[0], "GradientOf" + grad[1].name)
  minimizer = optimizer.apply_gradients(grads)
  merged = tf.merge_all_summaries()
  init = tf.initialize_all_variables()
  log_writer = tf.train.SummaryWriter(args.logdir + "/pong", graph)

  # Now we train.
  inputs, choices, results, rewards = [], [], [], []
  prev_input = None
  observation = env.reset()
  running_reward = -21.
  reward_sum = 0
  episode_number = 0
  to_take = 0

  session = tf.Session()
  with session.as_default():
    # print [x.name for x in graph.get_operations()]
    session.run(init)
    while True:
      if args.render: env.render()

      cur_input = prepro(observation)
      inp = cur_input - prev_input if prev_input is not None else np.zeros(D)
      prev_input = cur_input

      up_prob = session.run([end_val],
                            feed_dict={input_image : inp.reshape((1,D))})[0]

      action = 2 if np.random.uniform() > up_prob else 3
      choices.append(action - 2)
      inputs.append(inp)
      observation, reward, done, info = env.step(action)
      reward_sum += reward
      rewards.append(reward)

      if done:
        # Bookkeep
        running_reward = running_reward * 0.99 + reward_sum * 0.01
        print 'resetting env. episode reward total was %f. running mean %f' % (
                reward_sum, running_reward)
        reward_sum = 0
        observation = env.reset()
        episode_number += 1

        # Let's export statistics.
        xx = graph.get_tensor_by_name('FinalLayer/summaries/Max:0')
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summaries, _, _, x = session.run([merged, end_val, loss, xx],
                feed_dict = {
                    input_image : np.vstack(inputs[to_take:]),
                    rewards_ph : np.vstack([x for x in rewards[to_take:] if x]),
                    choices_ph : np.vstack(choices[to_take:]),
                    discounted_ph : discount_rewards(rewards[to_take:])},
                options=run_options,
                run_metadata=run_metadata)
        print x
        log_writer.add_run_metadata(run_metadata, 'Epis. %d' % episode_number)
        log_writer.add_summary(summaries, episode_number)
        to_take = len(rewards)

        # Learn.
        if (episode_number + 1) % args.batch_size == 0:
          feed_dict = {input_image: np.vstack(inputs),
                       choices_ph : np.vstack(choices),
                       discounted_ph : discount_rewards(rewards)}
          session.run(minimizer, feed_dict=feed_dict)
          inputs, choices, results, rewards = [], [], [], []
          to_take = 0

      if reward:
        print 'ep. %d game done, reward %f' % (episode_number, reward)
