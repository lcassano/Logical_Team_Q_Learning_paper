

# Import all packages

import numpy as np
from typing import Any, Optional, Sequence


class Replay(object):
  """Uniform replay buffer. Allocates all required memory at initialization."""

  def __init__(self, capacity: int, num_agents: int, obs_shape: int):
    """Initializes a new `Replay`.

    Args:
      capacity: The maximum number of items allowed in the replay.
    """
    self._data = None  # type: Optional[Sequence[np.ndarray]]
    self._capacity = capacity
    self._num_added = 0
    self._list_size = 0
    self._useless_samples = []
    self._is_useful = np.zeros(capacity, dtype=bool)

    self._data = {'old_obs': np.empty(shape=(self._capacity, num_agents, obs_shape), dtype=float),
                  'actions': np.empty(shape=(self._capacity, num_agents), dtype=int),
                  'rewards': np.empty(shape=self._capacity, dtype=float),
                  'discount': np.empty(shape=self._capacity, dtype=float),
                  'new_obs': np.empty(shape=(self._capacity, num_agents, obs_shape), dtype=float)}

  def add(self, new_data):
    """Adds data to the replay buffer."""

    transitions = new_data['rewards'].size

    for n in range(transitions):
      if self._list_size > 0:
        replace_idx = self._useless_samples.pop()
        self._is_useful[replace_idx] = True
        for (_, slot), (_, item) in zip(self._data.items(), new_data.items()):
          slot[replace_idx] = item[n]
        self._list_size -= 1
      else:
        for (_, slot), (_, item) in zip(self._data.items(), new_data.items()):
          idx = self._num_added % self._capacity
          slot[idx] = item[n]
          self._is_useful[idx] = True
        self._num_added += 1

  def sample(self, size: int) -> Sequence[np.ndarray]:
    """Returns a transposed/stacked minibatch. Each array has shape [B, ...]."""
    indices = np.random.randint(self.size, size=size)
    return indices, [value[indices] for (_, value) in self._data.items()]

  def reset(self,):
    """Resets the replay."""
    self._data = None

  @property
  def size(self) -> int:
    return min(self._capacity, self._num_added)

  @property
  def fraction_filled(self) -> float:
    return self.size / self._capacity

  def replace_list(self, sample_index):
    for k in range(np.size(sample_index)):
      if self._is_useful[sample_index[k]]:  #
        self._useless_samples.append(sample_index[k])
        self._list_size += 1
        self._is_useful[sample_index[k]] = False

  def __repr__(self):
    return 'Replay: size={}, capacity={}, num_added={}'.format(
        self.size, self._capacity, self._num_added)
