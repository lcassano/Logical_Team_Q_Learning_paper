

# Import all packages
import numpy as np
import matplotlib.pyplot as plt


class LTQL(object):
  """A simple implementation of tabular Logical Team Q-learning."""

  def __init__(
      self,
      test_obs,
      num_steps,
      nmbr_types,
      nmbr_agents_per_type,
      grid_length,
      agent_nmbr,
      n_actions_per_agent,
      obs_size,
      learning_rate,
      alpha,
      epsilon,
      discount,
      seed=None,
  ):

    self._q_b = {0: np.zeros(np.concatenate([obs_size[0], [n_actions_per_agent[0]]])),
                 1: np.zeros(np.concatenate([obs_size[1], [n_actions_per_agent[1]]]))}
    self._q_u = {0: np.zeros(np.concatenate([obs_size[0], [n_actions_per_agent[0]]])),
                 1: np.zeros(np.concatenate([obs_size[1], [n_actions_per_agent[1]]]))}

    # Hyperparameters.
    self._num_steps = num_steps
    self._nmbr_types = nmbr_types
    self._nmbr_agents_per_type = nmbr_agents_per_type
    self._grid_length = grid_length
    self._agent_nmbr = agent_nmbr
    self._n_actions_per_agent = n_actions_per_agent
    self.obs_size = obs_size
    self._learning_rate = learning_rate
    self._epsilon = epsilon
    self._discount = discount
    self._alpha = alpha
    self._steps_count = 0
    self._rng = np.random.RandomState(seed)
    self._plot_points = 1e3
    self._plot_period = int(np.max([num_steps//self._plot_points, 1]))
    self._q_history = np.empty([grid_length, num_steps//self._plot_period, n_actions_per_agent[1]])

  def policy(self, observations, exploit):
    """Select actions according to epsilon-greedy policy."""

    actions = np.empty([self._nmbr_types], dtype=int)

    for tau in range(self._nmbr_types):
      if not exploit and self._rng.rand() < np.maximum(0.05, self._epsilon * (1 - self._steps_count/20e4)):  #5e4
        actions[tau] = self._rng.randint(self._n_actions_per_agent[tau])
      else:
        index = tuple(observations[tau])
        actions[tau] = self._rng.choice(np.flatnonzero(self._q_b[tau][index] == np.max(self._q_b[tau][index])))

    return actions

  def update(self, observations, actions, reward, new_observations, last):
    if np.mod(self._steps_count, self._plot_period) == 0:  # End of episode
      for slot in range(self._grid_length):
        self._q_history[slot][self._steps_count // self._plot_period] = self._q_u[1][slot][1]
    self._steps_count += 1

    o = observations  # observations
    a = actions  # actions
    r = reward  # reward
    g = self._discount * float(1 - last)
    o_n = new_observations  # observations of new state
    optimal_plays = np.zeros(self._nmbr_types, dtype=bool)

    for tau in range(self._nmbr_types):
      q = self._q_b[tau][tuple(o[tau])]
      if q.max() == q[a[tau]]:
        optimal_plays[tau] = True

    for tau in range(self._nmbr_types):
      index = tuple(np.concatenate([o[tau], [a[tau]]]))
      q_b = self._q_b[tau][index]
      q_u = self._q_u[tau][index]
      q_next = np.max(self._q_u[tau][tuple(o_n[tau])])
      delta = r + g * q_next - q_b
      c1 = optimal_plays[1 - tau]
      c2 = delta > 0
      if c1:
        self._q_b[tau][index] = q_b + self._learning_rate * delta
        self._q_u[tau][index] = q_u + self._learning_rate * (r + g * q_next - q_u)
      elif c2:
        self._q_b[tau][index] = q_b + self._learning_rate * self._alpha * delta

  def plot_qs(self, save_path=''):
    """Plot results"""
    fig, ax = plt.subplots()
    plt.xlabel('Epochs', fontsize=25)
    plt.ylabel('Q function', fontsize=25)
    ax.plot(np.arange(start=0, stop=self._q_history.shape[1]*self._plot_period, step=self._plot_period),
            self._q_history[0], linewidth=2)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(['Stay', 'Left', 'Right'], fontsize=15)
    plt.grid()
    #plt.show()
    fig.savefig(fname=save_path+'logic_0', bbox_inches='tight')

    fig2, ax2 = plt.subplots()
    plt.xlabel('Epochs', fontsize=25)
    plt.ylabel('Q function', fontsize=25)
    ax2.plot(np.arange(start=0, stop=self._q_history.shape[1] * self._plot_period, step=self._plot_period),
            self._q_history[1], linewidth=2)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.legend(['Stay', 'Left', 'Right'], fontsize=15)
    plt.grid()
    #plt.show()
    fig2.savefig(fname=save_path+'logic_1', bbox_inches='tight')

    fig3, ax3 = plt.subplots()
    plt.xlabel('Epochs', fontsize=25)
    plt.ylabel('Q function', fontsize=25)
    ax3.plot(np.arange(start=0, stop=self._q_history.shape[1] * self._plot_period, step=self._plot_period),
            self._q_history[2], linewidth=2)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.legend(['Stay', 'Left', 'Right'], fontsize=15)
    plt.grid()
    #plt.show()
    fig3.savefig(fname=save_path+'logic_2', bbox_inches='tight')

    fig4, ax4 = plt.subplots()
    plt.xlabel('Epochs', fontsize=25)
    plt.ylabel('Q function', fontsize=25)
    ax4.plot(np.arange(start=0, stop=self._q_history.shape[1] * self._plot_period, step=self._plot_period),
            self._q_history[3], linewidth=2)
    ax4.tick_params(axis='both', which='major', labelsize=15)
    ax4.legend(['Stay', 'Left', 'Right'], fontsize=15)
    plt.grid()
    #plt.show()
    fig4.savefig(fname=save_path+'logic_3', bbox_inches='tight')

    plt.close('all')
