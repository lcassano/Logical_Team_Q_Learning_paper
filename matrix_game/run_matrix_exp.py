# Matrix Game experiment
# Author: Lucas Cassano
# Paper: "Logical Team Q-learning"
# ===================================

# Import necessary packages
from absl import app
from absl import flags
import numpy as np
import matplotlib.pyplot as plt
import q_mix

flags.DEFINE_string('save_path', '/tmp/', 'directory to save results.')


def main(argv):
  """Run simple 2 agent matrix game."""
  nmbr_games = 500
  seed = 1
  mu = 1e-1
  nmbr_agents = 2
  qmix_extra_iters = 100

  np.random.seed(seed)
  payoff = np.array([[0, 2, 0], [0, 1, 2]])  #np.array([[8, -12, -12], [-12, 0, 0], [-12, 0, 0]])  #
  q_joint = np.zeros_like(payoff)
  nmbr_actions_1, nmbr_actions_2 = payoff.shape
  q_logic_b = {0: np.zeros([nmbr_games, nmbr_actions_1]), 1: np.zeros([nmbr_games, nmbr_actions_2])}
  q_logic_u = {0: np.zeros([nmbr_games, nmbr_actions_1]), 1: np.zeros([nmbr_games, nmbr_actions_2])}
  q_dist = {0: np.zeros([nmbr_games, nmbr_actions_1]), 1: np.zeros([nmbr_games, nmbr_actions_2])}
  q_ind = {0: np.zeros([nmbr_games, nmbr_actions_1]), 1: np.zeros([nmbr_games, nmbr_actions_2])}
  q_tran = {0: np.zeros([nmbr_games, nmbr_actions_1]), 1: np.zeros([nmbr_games, nmbr_actions_2])}
  q_mix_class = q_mix.Qmix(payoff.shape, mu/2)
  q_mix_out = {0: np.zeros([nmbr_games, nmbr_actions_1]), 1: np.zeros([nmbr_games, nmbr_actions_2])}

  for n in range(nmbr_games - 1):
    actions = np.array([np.random.randint(nmbr_actions_1), np.random.randint(nmbr_actions_2)])  #Pick actions uniformly
    r = payoff[actions[0]][actions[1]]

    # Logic Team Q-learning
    for agent in range(nmbr_agents):
      q_logic_b[agent][n + 1] = q_logic_b[agent][n]
      q_logic_u[agent][n + 1] = q_logic_u[agent][n]
      chosen_action = actions[agent]
      if actions[nmbr_agents - 1 - agent] == np.argmax(q_logic_b[nmbr_agents - 1 - agent][n]):
        q_logic_b[agent][n + 1][chosen_action] += mu * (r - q_logic_b[agent][n][chosen_action])
        q_logic_u[agent][n + 1][chosen_action] += mu * (r - q_logic_u[agent][n][chosen_action])
      elif r > q_logic_b[agent][n][chosen_action]:
        q_logic_b[agent][n + 1][chosen_action] += mu * (r - q_logic_b[agent][n][chosen_action])

    # Independent Q-learning
    for agent in range(nmbr_agents):
      q_ind[agent][n + 1] = q_dist[agent][n]
      chosen_action = actions[agent]
      q_ind[agent][n + 1][chosen_action] += mu * (r - q_dist[agent][n][chosen_action])

    # Distributed Q-learning
    for agent in range(nmbr_agents):
      q_dist[agent][n + 1] = q_dist[agent][n]
      chosen_action = actions[agent]
      if r > q_dist[agent][n][chosen_action]:
        q_dist[agent][n + 1][chosen_action] += mu * (r - q_dist[agent][n][chosen_action])

    # Qtran-base
    q_joint[actions[0], actions[1]] -= (q_joint[actions[0], actions[1]] - r)
    q_j = q_joint[actions[0], actions[1]]
    q_tilde = q_tran[0][n][actions[0]] + q_tran[1][n][actions[1]]
    for agent in range(nmbr_agents):
      q_tran[agent][n + 1] = q_tran[agent][n]
      chosen_action = actions[agent]
      if q_tran[0][n][actions[0]] == np.max(q_tran[0][n]) and q_tran[1][n][actions[1]] == np.max(q_tran[1][n]):
        q_tran[agent][n + 1][chosen_action] -= mu * (q_tilde - q_j)
      else:
        q_tran[agent][n + 1][chosen_action] -= mu * np.minimum(q_tilde - q_j, 0)

    # Qmix
    q_mix_out[0][n + 1] = q_mix_out[0][n]
    q_mix_out[1][n + 1] = q_mix_out[1][n]
    for _ in range(qmix_extra_iters):  #Needs far extra iters to converge XD
      actions = np.array([np.random.randint(nmbr_actions_1), np.random.randint(nmbr_actions_2)])
      r = payoff[actions[0]][actions[1]]
      q1, q2, qmix = q_mix_class.learn(actions, r)
    q_mix_out[0][n + 1][actions[0]] = q1
    q_mix_out[1][n + 1][actions[1]] = q2

  # Print final Qmix matrices
  qmix1 = np.zeros([nmbr_actions_1])
  qmix2 = np.zeros([nmbr_actions_2])
  qmix_total = np.zeros([nmbr_actions_1, nmbr_actions_2])
  for a1 in range(nmbr_actions_1):
    for a2 in range(nmbr_actions_2):
      qmix1[a1], qmix2[a2], qmix_total[a1, a2] = q_mix_class.obtain_q([a1, a2])

  print(qmix1)
  print(qmix2)
  print(qmix_total)

  # Plot results
  fig1, ax1 = plt.subplots()
  plt.xlabel('Games', fontsize=25)
  plt.ylabel('Q-values', fontsize=25)
  ax1.plot(np.arange(start=0, stop=nmbr_games), q_logic_b[0], 'b')
  ax1.plot(np.arange(start=0, stop=nmbr_games), q_logic_b[1], 'r')
  ax1.set_yticks(np.arange(0, 2.01, step=0.5))
  ax1.tick_params(axis='both', which='major', labelsize=15)
  plt.grid()
  fig1.savefig(fname='biased_logic_matrix_game_1', bbox_inches='tight')

  fig2, ax2 = plt.subplots()
  plt.xlabel('Games', fontsize=25)
  plt.ylabel('Q-values', fontsize=25)
  ax2.plot(np.arange(start=0, stop=nmbr_games), q_logic_u[0], 'b')
  ax2.plot(np.arange(start=0, stop=nmbr_games), q_logic_u[1], 'r')
  ax2.set_yticks(np.arange(0, 2.01, step=0.5))
  ax2.tick_params(axis='both', which='major', labelsize=15)
  plt.grid()
  fig2.savefig(fname='unbiased_logic_matrix_game_1', bbox_inches='tight')

  fig3, ax3 = plt.subplots()
  plt.xlabel('Games', fontsize=25)
  plt.ylabel('Q-values', fontsize=25)
  ax3.plot(np.arange(start=0, stop=nmbr_games), q_dist[0], 'b')
  ax3.plot(np.arange(start=0, stop=nmbr_games), q_dist[1], 'r')
  ax3.set_yticks(np.arange(0, 2.01, step=0.5))
  ax3.tick_params(axis='both', which='major', labelsize=15)
  plt.grid()
  fig3.savefig(fname='q_dist_matrix_game_1', bbox_inches='tight')

  fig4, ax4 = plt.subplots()
  plt.xlabel('Games', fontsize=25)
  plt.ylabel('Q-values', fontsize=25)
  ax4.plot(np.arange(start=0, stop=nmbr_games * qmix_extra_iters, step=qmix_extra_iters), q_mix_out[0], 'b')
  ax4.plot(np.arange(start=0, stop=nmbr_games * qmix_extra_iters, step=qmix_extra_iters), q_mix_out[1], 'r')
  ax4.tick_params(axis='both', which='major', labelsize=15)
  plt.grid()
  fig4.savefig(fname='q_mix_matrix_game_1', bbox_inches='tight')

  fig5, ax5 = plt.subplots()
  plt.xlabel('Games', fontsize=25)
  plt.ylabel('Q-values', fontsize=25)
  ax5.plot(np.arange(start=0, stop=nmbr_games), q_ind[0], 'b')
  ax5.plot(np.arange(start=0, stop=nmbr_games), q_ind[1], 'r')
  ax5.set_yticks(np.arange(0, 2.01, step=0.5))
  ax5.tick_params(axis='both', which='major', labelsize=15)
  plt.grid()
  fig5.savefig(fname='ind_q_matrix_game_1', bbox_inches='tight')

  fig6, ax6 = plt.subplots()
  plt.xlabel('Games', fontsize=25)
  plt.ylabel('Q-values', fontsize=25)
  ax6.plot(np.arange(start=0, stop=nmbr_games), q_tran[0], 'b')
  ax6.plot(np.arange(start=0, stop=nmbr_games), q_tran[1], 'r')
  ax6.set_yticks(np.arange(0, 2.01, step=0.5))
  ax6.tick_params(axis='both', which='major', labelsize=15)
  plt.grid()
  fig6.savefig(fname='q_tran_matrix_game_1', bbox_inches='tight')
  print(np.expand_dims(q_tran[0][-1], axis=1))
  print(np.expand_dims(q_tran[1][-1], axis=0))
  print(np.expand_dims(q_tran[0][-1], axis=1) + np.expand_dims(q_tran[1][-1], axis=0))

  return 1


if __name__ == '__main__':
  app.run(main)
