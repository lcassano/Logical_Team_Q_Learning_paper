# coding=utf-8

# Import all packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def run(agent,
        environment,
        num_agents: int,
        num_epochs: int,
        nmbr_games: int,
        save_period: int,
        seed: int,
        save_model: bool,
        save_path):
  """Runs the experiment.
  """
  tests = 50
  c1_c2_games_test = tf.constant([0, tests * num_agents], shape=(2, ), dtype=tf.int32)
  obs_features = environment.observation_shape()[0]

  test_number = 0
  nmbr_tests = 500
  test_step = np.maximum(200, int(np.ceil(num_epochs / nmbr_tests)))
  num_samples = int(np.ceil(num_epochs / test_step))
  test_returns = np.zeros(num_samples)
  percentage_games_won = np.zeros(num_samples)
  for n in range(num_epochs):

    # Test performance
    if n % test_step == 0:
      exploit = tf.constant(True)
      cum_r = np.zeros(tests)
      obs, discount = environment.reset(tests)
      while any(discount == 1.0):
        tensor_obs = tf.constant(np.reshape(obs, newshape=(tests * num_agents, obs_features)), dtype=tf.float32)
        actions = agent.policy(tensor_obs, exploit, c1_c2_games_test)
        reward, obs, discount = environment.step(np.reshape(actions, newshape=(tests, num_agents)))
        cum_r += reward
      games_won = np.sum(cum_r > 0)
      test_returns[test_number] = np.mean(cum_r)
      percentage_games_won[test_number] = games_won / tests
      test_number += 1
      print('-----------------------------------------------------------')
      print('Epoch: %d. Avg performance: %f. Games won: %d/%d' % (n, np.mean(cum_r), games_won, tests))


    # Proceed to collect data.
    obs, discount = environment.reset(nmbr_games)
    active_games = discount.astype(dtype=bool)
    all_actions = np.zeros((nmbr_games, num_agents), dtype=int)
    while any(discount == 1.0):
      unfinished_games = np.sum(active_games)
      tensor_obs = tf.constant(np.reshape(obs[active_games], newshape=(unfinished_games * num_agents, obs_features)), dtype=tf.float32)
      number_c1_games = np.sum(active_games[0: int(np.floor(nmbr_games/2))])
      c1_c2_games = tf.constant([number_c1_games * num_agents, (unfinished_games - number_c1_games) * num_agents], shape=(2,), dtype=tf.int32)
      actions = agent.policy(obs=tensor_obs, exploit=tf.constant(False), c1_c2_games=c1_c2_games)

      all_actions[active_games] = np.reshape(actions, newshape=(unfinished_games, num_agents))
      reward, new_obs, discount = environment.step(all_actions)

      agent.store_data(obs, all_actions, reward, discount, new_obs, active_games)  # Save data in replay buffer.

      active_games = np.logical_and(active_games, discount.astype(dtype=bool))
      obs = new_obs

    agent.update()  # Learn

    if n % save_period == 0:  #Make partial plots
      np.savetxt(fname=save_path + 'percentage_games_won_seed_%d.csv' % seed, X=percentage_games_won, delimiter=',')
      #agent.plot_qs(show=False)

      fig, ax = plt.subplots()
      plt.xlabel('Epochs', fontsize=25)
      plt.ylabel('Avg return', fontsize=25)
      ax.plot(np.arange(start=0, stop=test_returns.shape[0] * test_step, step=test_step), test_returns)
      plt.grid()
      fig.savefig(fname=save_path + 'returns')

      fig, ax = plt.subplots()
      plt.xlabel('Epochs', fontsize=25)
      plt.ylabel('Test win %', fontsize=25)
      ax.plot(np.arange(start=0, stop=test_returns.shape[0] * test_step, step=test_step), percentage_games_won)
      plt.grid()
      fig.savefig(fname=save_path + 'games_won')

      plt.close('all')
      if save_model:
        agent.save_model()

  #agent.plot_qs(show=False)
  fig, ax = plt.subplots()
  plt.xlabel('Epochs', fontsize=25)
  plt.ylabel('Avg return', fontsize=25)
  ax.plot(np.arange(start=0, stop=test_returns.shape[0] * test_step, step=test_step), test_returns)
  plt.grid()
  fig.savefig(fname=save_path + 'returns', bbox_inches='tight')

  fig1, ax1 = plt.subplots()
  plt.xlabel('Epochs', fontsize=25)
  plt.ylabel('Test win %', fontsize=25)
  ax1.plot(np.arange(start=0, stop=test_returns.shape[0] * test_step, step=test_step), percentage_games_won)
  plt.grid()
  fig1.savefig(fname=save_path + 'games_won', bbox_inches='tight')
  plt.close('all')

  if save_model:
    agent.save_model()

  return percentage_games_won, test_returns


def test(agent, environment, num_agents: int, nmbr_games=1):
  """Tests policy on environment."""
  obs_features = environment.observation_shape()[0]

  exploit = tf.constant(True)
  games_won = np.zeros(nmbr_games, dtype=bool)
  obs, discount = environment.reset(nmbr_games)
  c1_c2_games_test = tf.constant([0, nmbr_games * num_agents], shape=(2, ), dtype=tf.int32)
  while any(discount == 1.0):
    tensor_obs = tf.constant(np.reshape(obs, newshape=(nmbr_games * num_agents, obs_features)), dtype=tf.float32)
    actions = agent.policy(tensor_obs, exploit, c1_c2_games_test)
    reward, obs, discount = environment.step(np.reshape(actions, newshape=(nmbr_games, num_agents)))
    games_won = np.logical_or(games_won, reward > 0)

  return np.sum(games_won) / nmbr_games
