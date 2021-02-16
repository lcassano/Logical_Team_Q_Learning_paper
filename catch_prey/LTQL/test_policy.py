# Script to test pre-trained policy.
# Author: Lucas Cassano
# Paper: ""Logical Team Q-learning: An approach towards optimal factored policies in cooperative MARL""
# ======================================================================================================================

# Import all packages

import os
os.sys.path.append("/tmp/Logical Team Q learning/experiments/")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #Turn off GPU

from absl import app
from absl import flags

from catch_prey.catch_prey import CatchPrey
import ltql

import tensorflow as tf
import numpy as np

flags.DEFINE_boolean('make_videos', True, 'Generate videos after training.')
flags.DEFINE_string('path', 'results/1/', 'where to load model and save results')

flags.DEFINE_integer('agent_number', 4, 'number of agents.')
flags.DEFINE_integer('number_games', 4, 'number of games to play (one video per game will be made).')

# Network architecture hyper-parameters
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 50, 'number of units per hidden layer')


FLAGS = flags.FLAGS


def main(argv):
  """Runs a Logical Team Q-learning agent on the catch-prey game."""

  device = '/'

  env = CatchPrey(agent_nmbr=FLAGS.agent_number)  # Load the environment.
  q_bound = np.max([env.max_return(0.99), np.abs(env.min_return(0.99))])
  data_spec = {'float': tf.float32, 'int': tf.int32}

  path = FLAGS.path
  nmbr_agents = FLAGS.agent_number
  agent = ltql.LTQL(
        test_observations=env.get_debug_observations(),
        nmbr_episodes=1,
        max_schedule=1,
        nmbr_agents=nmbr_agents,
        obs_shape=env.observation_shape(),
        action_spec=env.action_space_size(),
        q_bound=q_bound,
        num_units=FLAGS.num_units,
        num_hidden_layers=FLAGS.num_hidden_layers,
        batch_size=1,
        discount=1,
        temp_max=1,
        temp_min=1,
        replay_capacity=1,
        min_replay_size=1,
        target_update_period=1,
        optimizer_q=tf.keras.optimizers.Adam(learning_rate=1),
        optimizer_opt_q=tf.keras.optimizers.Adam(learning_rate=1),
        step_size_ratio=1,
        learn_iters=1,
        load_weights=True,
        data_spec=data_spec,
        device=device,
        seed=0,
        save_path=path
  )

  # Test performance of learnt strategy.
  number_games = FLAGS.number_games
  obs_features = env.observation_shape()[0]
  c1_c2_games_test = tf.constant([0, number_games * nmbr_agents], shape=(2,), dtype=tf.int32)
  exploit = tf.constant(True)
  cum_r = np.zeros(number_games)
  obs, discount = env.reset(parallel_games=number_games)
  if FLAGS.make_videos:
    env.init_videos(path)
  while any(discount == 1.0):
    tensor_obs = tf.constant(np.reshape(obs, newshape=(number_games * nmbr_agents, obs_features)), dtype=tf.float32)
    actions = agent.policy(tensor_obs, exploit, c1_c2_games_test)
    reward, obs, discount = env.step(np.reshape(actions, newshape=(number_games, nmbr_agents)))
    cum_r += reward
    if FLAGS.make_videos:
      env.add_frames(reward)
  games_won = np.sum(cum_r > 0)

  print('Games won: %d out of %d' % (games_won, number_games))
  return 1


if __name__ == '__main__':
  app.run(main)
