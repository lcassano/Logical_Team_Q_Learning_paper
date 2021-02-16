# coding=utf-8
# Copyright 2020 Lucas Cassano.

"""TF2 Logical Team Q-learning Implementation."""

# Import all packages
from catch_prey.utils import batched_index, MyNNs
from tensorflow.keras.layers import Dense, Lambda
from catch_prey import replay2
import numpy as np
import tensorflow as tf


class Hysteretic(object):
  """A TensorFlow2 implementation of Logical Team Q-learning."""

  def __init__(
    self,
    test_observations,
    nmbr_episodes,
    max_schedule,
    nmbr_agents,
    obs_shape,
    action_spec,
    q_bound,
    num_units,
    num_hidden_layers,
    batch_size,
    discount,
    temp_max,
    temp_min,
    replay_capacity,
    min_replay_size,
    target_update_period,
    optimizer_q,
    step_size_ratio,
    learn_iters,
    load_weights,
    data_spec,
    device='cpu:*',
    seed=None,
    save_path=''
  ):

    self._save_path = save_path
    self._device = device
    self._optimizer_q = optimizer_q
    # Hyperparameters.
    self._nmbr_agents = nmbr_agents
    self._batch_size = batch_size
    self._batch_size_tf = tf.constant(batch_size, dtype=tf.float32)
    self._num_actions = action_spec
    self._learning_updates = learn_iters
    self._target_update_period = target_update_period
    self._gamma = discount
    self._step_size_ratio = step_size_ratio
    self._total_steps = 0
    self._eps_count = 0
    self._max_schedule = max_schedule
    self._replay = replay2.Replay(capacity=replay_capacity, num_agents=nmbr_agents, obs_shape=obs_shape[0])
    self._min_replay_size = min_replay_size
    self._temp = tf.Variable(temp_max, dtype=tf.float32)
    self._temp_min = tf.constant(temp_min, dtype=tf.float32)
    self._temp_max = tf.constant(temp_max, dtype=tf.float32)
    self._learn_iter_counter = 0

    q_out_fn = Lambda(lambda x: q_bound * tf.tanh(x))
    self._q_net = MyNNs(True, obs_shape, num_hidden_layers, num_units, action_spec, 'zeros', q_out_fn)
    self._q_target_net = MyNNs(True, obs_shape, num_hidden_layers, num_units, action_spec, 'zeros', q_out_fn)

    if load_weights:
      self.load_model()

  # @tf.function(input_signature=[
  #       tf.TensorSpec(shape=[None, 3, 7], dtype=tf.float32),
  #       tf.TensorSpec(shape=[None, 3], dtype=tf.int32),
  #       tf.TensorSpec(shape=[None, ], dtype=tf.float32),
  #       tf.TensorSpec(shape=[None, ], dtype=tf.float32),
  #       tf.TensorSpec(shape=[None, 3, 7], dtype=tf.float32)])
  @tf.function
  def _learn(self, o_t, a_t, r_t, d_tp1, o_tp1):
    with tf.GradientTape() as tape:
      tape.watch(self._q_net.trainable_weights)

      my_obs = o_t[:, 0, :]
      my_new_obs = o_tp1[:, 0, :]

      # Q-network loss
      q_t = self._q_net(my_obs, training=True)
      train_q_value = batched_index(q_t, a_t[:, 0])
      q_tp1 = tf.stop_gradient(tf.reduce_max(self._q_target_net(my_new_obs), axis=-1))
      q_target_value = tf.stop_gradient(r_t + self._gamma * d_tp1 * q_tp1)
      delta_q = q_target_value - train_q_value

      c2 = tf.stop_gradient(tf.math.greater(delta_q, 0))
      step_size_scale = tf.stop_gradient(tf.cast(c2, q_target_value.dtype)) + self._step_size_ratio * tf.stop_gradient(tf.cast(tf.math.logical_not(c2), q_target_value.dtype))
      loss_q = step_size_scale * tf.square(delta_q, name='loss_q')

    q_variables_to_train = self._q_net.trainable_weights
    q_grads = tape.gradient(loss_q, q_variables_to_train)
    self._optimizer_q.apply_gradients(list(zip(q_grads, q_variables_to_train)))

  @tf.function
  def _update_target_nets(self):
    with tf.device(self._device):
      source_variables = self._q_net.trainable_weights
      target_variables = self._q_target_net.trainable_weights
      for (v_s, v_t) in zip(source_variables, target_variables):
        v_t.shape.assert_is_compatible_with(v_s.shape)
        v_t.assign(v_s)

  @tf.function(input_signature=[tf.TensorSpec(shape=(None, 7), dtype=tf.float32)])
  def _q_target_fn(self, obs):
    return self._q_target_net(obs, training=False)

  @tf.function(input_signature=[tf.TensorSpec(shape=(None, 7), dtype=tf.float32)])
  def _q_fn(self, obs):
    return self._q_net(obs)

  @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
        tf.TensorSpec(shape=( ), dtype=tf.bool),
        tf.TensorSpec(shape=(2, ), dtype=tf.int32)])
  def policy(self, obs, exploit, c1_c2_games):
    """Select actions according to epsilon-greedy policy."""

    @tf.function
    def exploratoy_policy(temp):
      q_values = self._q_target_net(obs)
      obs_list = tf.split(q_values, num_or_size_splits=c1_c2_games, axis=0, num=None, name='split')
      explore = tf.squeeze(tf.random.categorical(obs_list[0] / temp, 1, dtype=tf.int32))
      greedy = tf.squeeze(tf.argmax(obs_list[1], axis=-1, output_type=tf.int32))
      return tf.concat([explore, greedy], axis=0)

    @tf.function
    def greedy_policy():
      q_values = self._q_target_net(obs)
      return tf.squeeze(tf.argmax(q_values, axis=-1, output_type=tf.int32))

    if exploit:
      actions = greedy_policy()
    else:
      actions = exploratoy_policy(temp=tf.math.maximum(self._temp_min, self._temp_max * (1 - self._eps_count/self._max_schedule)))

    return actions

  def store_data(self, obs, actions, reward, discount, new_obs, active_games):
    """Stores new data in the replay buffer."""

    new_data = {'old_obs': obs[active_games],
                'actions': actions[active_games],
                'rewards': reward[active_games],
                'discount': discount[active_games],
                'new_obs': new_obs[active_games]}

    for a in range(self._nmbr_agents):
      self._replay.add(new_data)
      new_data['old_obs'] = np.roll(new_data['old_obs'], -1, axis=1)
      new_data['new_obs'] = np.roll(new_data['new_obs'], -1, axis=1)
      new_data['actions'] = np.roll(new_data['actions'], -1, axis=1)

  #@tf.function
  def update(self):
    """Takes in a transition from the environment."""

    self._eps_count += 1
    if self._replay.size >= self._min_replay_size:
      for _ in range(self._learning_updates):
        samples_indices, minibatch = self._replay.sample(self._batch_size)
        tf_minibatch = [tf.constant(mat, dtype=tf_type) for mat, tf_type in zip(minibatch, [tf.float32, tf.int32, tf.float32, tf.float32, tf.float32])]
        self._learn(*tf_minibatch)

        self._learn_iter_counter += 1
        if (self._target_update_period > 1) and (self._learn_iter_counter % self._target_update_period == 0):
          self._update_target_nets()

  def load_model(self):
    """load all networks"""
    self._q_net.load_weights(self._save_path + 'q_net.h5')
    self._q_target_net.load_weights(self._save_path + 'q_target_net.h5')

  def save_model(self):
    """Save all networks"""
    self._q_net.save_weights(self._save_path + 'q_net.h5')
    self._q_target_net.save_weights(self._save_path + 'q_target_net.h5')
