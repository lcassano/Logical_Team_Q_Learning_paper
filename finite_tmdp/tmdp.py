
# Import required packages
import numpy as np


class TeamMDP:
  """Stochastic Team MDP"""

  def __init__(self, seed=None):
    """
    Args:
      seed: Optional integer. Seed for numpy's random number generator.
    """
    self._rng = np.random.RandomState(seed)
    self.obs_length = 3
    self.nmbr_types = 2
    self.agent_nmbr_per_type = 1
    self.agent_nmbr = self.nmbr_types * self.agent_nmbr_per_type
    self.grid_length = 4
    self._move_r = 0
    self._illegal_press = -30
    self._success = 10
    self._not_press_penalty = -30
    self._time_reward = 0
    self._fixed_noise_std_dev = 1.0
    self._std_dev = 3.0
    self._time_limit = 4
    self._time = 0

    self.n_actions_per_agent = np.array([2, 3])  # First type only press or not, the other: stay-left-right
    self._n_actions = self.n_actions_per_agent[0] * self.n_actions_per_agent[1]
    self._opt_s = np.array([self.grid_length - 1, 0])  #Postions in which if both jump they win
    self._opt_a = np.array([1, 0])  #Everybody goes up.
    self._current_states = np.array([self.grid_length - 1, self.grid_length - 1])  # Always start in the far right

  def reset(self):
    self._current_states = np.array([self.grid_length - 1, self.grid_length - 1])  # Always start in the far right
    self._time = 0
    return self._get_observation()

  def step(self, actions):
    clean_reward = 0
    reward_noise = self._fixed_noise_std_dev * self._rng.randn()
    self._time += 1
    last = False

    if np.all(self._current_states == self._opt_s) and np.all(actions == self._opt_a):  # Won
      clean_reward = self._success
      last = True
    elif np.all(self._current_states == self._opt_s) and actions[0] == 0 and actions[1] == 0:
      clean_reward = self._not_press_penalty
      last = True
    elif actions[1] == 0:  #Do nothing
      clean_reward = 0
    elif actions[1] == 1:  #Wants to move left
      if self._current_states[1] > 0:
        if actions[0] == 1:  # Bad press!
          self._current_states[1] -= 1  #Move
          clean_reward = self._illegal_press
        else:
          self._current_states[1] -= 1  #Move left
          clean_reward = self._move_r
      else:
        reward_noise = self._std_dev * self._rng.randn()
    elif actions[1] == 2:
      if self._current_states[1] < self.grid_length - 1:
        clean_reward = self._move_r
        self._current_states[1] += 1  #Move right
      else:
        clean_reward = 0
        reward_noise = self._std_dev * self._rng.randn()
    else:
      print("Move error")

    clean_reward += self._time_reward
    reward = clean_reward + reward_noise

    if self._time >= self._time_limit:
      last = True
    return reward, self._get_observation(), last, clean_reward

  def _get_state(self):
    return self._current_states

  def action_space_size(self):
    return self.n_actions_per_agent

  def obs_space_size(self):
    obs_size = {0: [2] + [2],
                1: [self.grid_length] + [2]}
    return obs_size

  def _get_observation(self, states=None, time=None):
    if states is None:
      states = self._current_states
    if time is None:
      time = self._time

    observations = {0: np.array([int(states[1] == self._opt_s[1]), int(self._time_limit - time > states[1])]),  # Size 2
                    1: np.array([states[1], int(self._time_limit - time > states[1])])}  # Size self.grid_length x 2
    return observations

  def get_opt_oa(self):
    return self._get_observation(self._opt_s, 5)

  def max_return(self):
    return self._success + (self.grid_length - 1) * self._move_r
