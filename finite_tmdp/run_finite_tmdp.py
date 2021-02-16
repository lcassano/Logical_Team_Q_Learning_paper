# Finite TMDP experiment
# Author: Lucas Cassano
# Paper: "Logical Team Q-learning"
# ======================================================================================================================

# Import necessary packages
from absl import app
from absl import flags
from tmdp import TeamMDP
import LTQL
import dist_q
import hyst_q
import q_tran
import q_mix
import numpy as np
import matplotlib.pyplot as plt


flags.DEFINE_string('save_path', '/Users/lucas/Desktop/', 'directory to save results.')
flags.DEFINE_integer('num_steps', 100000, 'number of training episodes.')  #30000
flags.DEFINE_float('learning_rate', 2.5e-2, 'q learning rate.') #1e-1
flags.DEFINE_float('small_learning_rate', 1e-2, 'small learning rate for hysteretic Q-learning.')
flags.DEFINE_float('epsilon', 1, 'fraction of exploratory random actions')
flags.DEFINE_float('discount', .99, 'discounting on the agent side')
flags.DEFINE_float('alpha', 1.0, '''LTQL's alpha parameter''')
flags.DEFINE_integer('number_seeds', 20, 'seed for random number generation')

FLAGS = flags.FLAGS

colors = {'LTQL': 'b', 'Qtran': 'r', 'HystQ': 'g', 'DistQ': 'c', 'Qmix': 'y', 'IQL': 'm'}


def main(argv):
  """Run experiment."""
  del argv  # Unused.

  num_steps = FLAGS.num_steps
  plot_points = 1e2
  test_period = int(np.max([num_steps // plot_points, 1]))
  ep_return = np.zeros([num_steps // test_period, colors.__len__(), FLAGS.number_seeds])
  test_win_rate = np.zeros([num_steps // test_period, colors.__len__(), FLAGS.number_seeds])

  for seed in range(FLAGS.number_seeds):
    # Load the environment.
    env = TeamMDP(seed=seed)
    logical_agent = LTQL.LTQL(
        test_obs=env.get_opt_oa(),
        num_steps=FLAGS.num_steps,
        nmbr_types=env.nmbr_types,
        nmbr_agents_per_type=env.agent_nmbr_per_type,
        grid_length=env.grid_length,
        agent_nmbr=env.agent_nmbr,
        n_actions_per_agent=env.n_actions_per_agent,
        obs_size=env.obs_space_size(),
        learning_rate=FLAGS.learning_rate,
        alpha=FLAGS.alpha,
        epsilon=FLAGS.epsilon,
        discount=FLAGS.discount,
        seed=seed,
      )
    hyst_agent = hyst_q.HystQ(
      test_obs=env.get_opt_oa(),
      num_steps=FLAGS.num_steps,
      nmbr_types=env.nmbr_types,
      nmbr_agents_per_type=env.agent_nmbr_per_type,
      grid_length=env.grid_length,
      agent_nmbr=env.agent_nmbr,
      n_actions_per_agent=env.n_actions_per_agent,
      obs_size=env.obs_space_size(),
      learning_rate=FLAGS.learning_rate,
      small_learning_rate=FLAGS.small_learning_rate,
      epsilon=FLAGS.epsilon,
      discount=FLAGS.discount,
      seed=seed,
    )
    ind_agent = hyst_q.HystQ(
      test_obs=env.get_opt_oa(),
      num_steps=FLAGS.num_steps,
      nmbr_types=env.nmbr_types,
      nmbr_agents_per_type=env.agent_nmbr_per_type,
      grid_length=env.grid_length,
      agent_nmbr=env.agent_nmbr,
      n_actions_per_agent=env.n_actions_per_agent,
      obs_size=env.obs_space_size(),
      learning_rate=FLAGS.learning_rate,
      small_learning_rate=FLAGS.learning_rate,
      epsilon=FLAGS.epsilon,
      discount=FLAGS.discount,
      seed=seed,
    )
    dist_q_agent = dist_q.DistQ(
      test_obs=env.get_opt_oa(),
      num_steps=FLAGS.num_steps,
      nmbr_types=env.nmbr_types,
      nmbr_agents_per_type=env.agent_nmbr_per_type,
      grid_length=env.grid_length,
      agent_nmbr=env.agent_nmbr,
      n_actions_per_agent=env.n_actions_per_agent,
      obs_size=env.obs_space_size(),
      learning_rate=FLAGS.learning_rate,
      epsilon=FLAGS.epsilon,
      discount=FLAGS.discount,
      seed=seed,
    )
    qtran_agent = q_tran.Qtran(
      test_obs=env.get_opt_oa(),
      num_steps=FLAGS.num_steps,
      nmbr_types=env.nmbr_types,
      nmbr_agents_per_type=env.agent_nmbr_per_type,
      grid_length=env.grid_length,
      agent_nmbr=env.agent_nmbr,
      n_actions_per_agent=env.n_actions_per_agent,
      obs_size=env.obs_space_size(),
      learning_rate=FLAGS.learning_rate,
      epsilon=FLAGS.epsilon,
      discount=FLAGS.discount,
      seed=seed,
    )
    qmix_agent = q_mix.Qmix(
        test_obs=env.get_opt_oa(),
        num_steps=FLAGS.num_steps,
        nmbr_types=env.nmbr_types,
        nmbr_agents_per_type=env.agent_nmbr_per_type,
        grid_length=env.grid_length,
        agent_nmbr=env.agent_nmbr,
        n_actions_per_agent=env.n_actions_per_agent,
        obs_size=env.obs_space_size(),
        learning_rate=FLAGS.learning_rate,
        epsilon=FLAGS.epsilon,
        discount=FLAGS.discount,
        seed=seed,
    )

    agents = {'LTQL': logical_agent, 'Qtran': qtran_agent, 'HystQ': hyst_agent, 'DistQ': dist_q_agent, 'Qmix': qmix_agent, 'IQL': ind_agent}
    k = 0
    number_tests = 50
    for _, agent in agents.items():
      last = True
      for n in range(num_steps):
        if np.mod(n, test_period) == 0:
          for _ in range(number_tests):
            observations = env.reset()
            last = False
            while not last:
              action = agent.policy(observations, exploit=True)
              _, new_observations, last, clean_reward = env.step(action)
              observations = new_observations
              ep_return[n//test_period, k, seed] += clean_reward
            test_win_rate[n//test_period, k, seed] += ep_return[n//test_period, k, seed] > 0

        if last:
          observations = env.reset()
        action = agent.policy(observations, exploit=False)
        reward, new_observations, last, _ = env.step(action)
        agent.update(observations, action, reward, new_observations, last)
        observations = new_observations

      agent.plot_qs(save_path=FLAGS.save_path)
      k += 1

  ep_return = ep_return / number_tests
  test_win_rate = test_win_rate / number_tests * 100

  fig, ax = plt.subplots()
  plt.xlabel('Epochs (thousands)', fontsize=25)
  plt.ylabel('Test win %', fontsize=25)
  k = 0
  my_legend = []
  for agent in agents:
    x = np.arange(start=0, stop=ep_return.shape[0] * test_period, step=test_period) / 1000
    ax.plot(x, np.mean(test_win_rate[:, k, :], axis=1), linewidth=4.5-0.5*k, color=colors[agent])
    y_min = np.min(test_win_rate[:, k, :], axis=1)
    y_max = np.max(test_win_rate[:, k, :], axis=1)
    ax.fill_between(x, y_min, y_max, alpha=0.3, color=colors[agent])
    my_legend.append(agent)
    k += 1
  ax.tick_params(axis='both', which='major', labelsize=15)
  ax.legend(my_legend, fontsize=15, loc='upper right', bbox_to_anchor=(0.95, 0.65))
  plt.grid()
  # plt.show()
  fig.savefig(fname=FLAGS.save_path + 'test_win_rate_TMDP', bbox_inches='tight')

  fig2, ax2 = plt.subplots()
  plt.xlabel('Epochs (thousands)', fontsize=25)
  plt.ylabel('Average test return', fontsize=25)
  k = 0
  for agent in agents:
    x = np.arange(start=0, stop=ep_return.shape[0] * test_period, step=test_period) / 1000
    ax2.plot(x, np.mean(ep_return[:, k, :], axis=1), linewidth=5.25-0.75*k, color=colors[agent])
    y_min = np.min(ep_return[:, k, :], axis=1)
    y_max = np.max(ep_return[:, k, :], axis=1)
    ax2.fill_between(x, y_min, y_max, alpha=0.3, color=colors[agent])
    k += 1
  ax2.tick_params(axis='both', which='major', labelsize=15)
  plt.ylim(-20, 12)
  ax2.legend(my_legend, fontsize=15, loc='upper right', bbox_to_anchor=(0.95, 0.55))
  plt.grid()
  plt.ylim(bottom=-10)
  #plt.show()
  fig2.savefig(fname=FLAGS.save_path+'performance_TMDP', bbox_inches='tight')

  return 1


if __name__ == '__main__':
  app.run(main)
