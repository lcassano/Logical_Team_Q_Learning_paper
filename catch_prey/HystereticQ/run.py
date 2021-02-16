# Hysteretic Q-learning implementation
# Author: Lucas Cassano
# Paper: "Logical Team Q-learning: An approach towards optimal factored policies in cooperative MARL"
# ======================================================================================================================

# Import all packages

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #Turn off GPU

from absl import app
from absl import flags

from .. import experiment
from ..catch_prey import CatchPrey
from . import hysteretic

import tensorflow as tf
import numpy as np


flags.DEFINE_boolean('save', True, 'save params and results.')
flags.DEFINE_boolean('load_weights', False, 'Load the weights of the networks.')
flags.DEFINE_boolean('save_model', True, 'save the weights of the networks.')
flags.DEFINE_integer('save_period', 1000, 'Save partial plots and learnt models every --save_period epochs.')
flags.DEFINE_string('save_path', '/tmp/', 'directory to save results')

flags.DEFINE_integer('num_epochs', 100000, 'number of training epochs.')
flags.DEFINE_integer('agent_number', 4, 'number of agents.')
flags.DEFINE_integer('seed', 0, 'initial seed')

# Network architecture hyper-parameters
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 50, 'number of units per hidden layer')

# Hysteretic hyper-parameters
flags.DEFINE_integer('batch_size', 1024, 'mini-batch size')
flags.DEFINE_integer('nmbr_games', 32, 'Amount of transitions to gather at each interaction with the env.')
flags.DEFINE_float('agent_discount', 0.99, 'discounting to reduce variance')
flags.DEFINE_integer('replay_capacity', 200000, 'size of the replay buffer')
flags.DEFINE_integer('min_replay_size', 2048, 'min amount of transitions before sampling')
flags.DEFINE_integer('target_update_period', 50, 'steps between target net updates')
flags.DEFINE_float('learning_rate', 1e-5, 'learning rate')
flags.DEFINE_float('step_size_ratio', 0.1, 'ratios of small to big step size')
flags.DEFINE_integer('learn_iters', 50, 'gradient descent iterations to optimize nets')

# Behavior policy hyperparameters
flags.DEFINE_float('temp_max', 0.5, 'Initial temperature parameter for Boltzmann policy.')
flags.DEFINE_float('temp_min', 0.05, 'Final temperature parameter for Boltzmann policy after annealing.')
flags.DEFINE_float('max_schedule', 15e3, 'Final temperature parameter for Boltzmann policy after annealing.')


FLAGS = flags.FLAGS


def main(argv):
  """Runs a Hysteretic Q-learning agent on the catch-prey game."""

  if FLAGS.save:
    if not os.path.exists(FLAGS.save_path):
      os.mkdir(FLAGS.save_path)
    f = open(FLAGS.save_path + "experiment_parameters.txt", "w+")
    for item in FLAGS.__dict__['__flags']:
      item_value = FLAGS.__dict__['__flags'][item]._value
      if isinstance(item_value, int):
        f.write(item + ' = %d\r\n' % item_value)
      if isinstance(item_value, float):
        f.write(item + ' = %.6f\r\n' % item_value)
      elif isinstance(item_value, str):
        f.write(item + ' = ' + item_value + '\r\n')
    f.write('save_path = ' + FLAGS.save_path)
    f.close()

  print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
  print(tf.config.list_physical_devices())

  device = '/'
  # tf.debugging.set_log_device_placement(True)

  env = CatchPrey(agent_nmbr=FLAGS.agent_number, seed=FLAGS.seed)  # Load the environment.
  q_bound = np.max([env.max_return(FLAGS.agent_discount), np.abs(env.min_return(FLAGS.agent_discount))])
  data_spec = {'float': tf.float32, 'int': tf.int32}

  agent = hysteretic.Hysteretic(
        test_observations=env.get_debug_observations(),
        nmbr_episodes=FLAGS.num_epochs,
        max_schedule=FLAGS.max_schedule,
        nmbr_agents=FLAGS.agent_number,
        obs_shape=env.observation_shape(),
        action_spec=env.action_space_size(),
        q_bound=q_bound,
        num_units=FLAGS.num_units,
        num_hidden_layers=FLAGS.num_hidden_layers,
        batch_size=FLAGS.batch_size,
        discount=FLAGS.agent_discount,
        temp_max=FLAGS.temp_max,
        temp_min=FLAGS.temp_min,
        replay_capacity=FLAGS.replay_capacity,
        min_replay_size=FLAGS.min_replay_size,
        target_update_period=FLAGS.target_update_period,
        optimizer_q=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
        step_size_ratio=FLAGS.step_size_ratio,
        learn_iters=FLAGS.learn_iters,
        load_weights=FLAGS.load_weights,
        data_spec=data_spec,
        device=device,
        seed=FLAGS.seed,
        save_path=FLAGS.save_path
  )

  percentage_games_won, returns = experiment.run(agent=agent,
                                                 environment=env,
                                                 num_agents=FLAGS.agent_number,
                                                 num_epochs=FLAGS.num_epochs,
                                                 nmbr_games=FLAGS.nmbr_games,
                                                 save_period=FLAGS.save_period,
                                                 seed=FLAGS.seed,
                                                 save_model=FLAGS.save_model,
                                                 save_path=FLAGS.save_path)

  np.savetxt(fname=FLAGS.save_path+'percentage_games_won_seed_%d.csv' % FLAGS.seed, X=percentage_games_won, delimiter=',')
  np.savetxt(fname=FLAGS.save_path + 'returns_seed_%d.csv' % FLAGS.seed, X=returns, delimiter=',')

  return 1


if __name__ == '__main__':
  app.run(main)
