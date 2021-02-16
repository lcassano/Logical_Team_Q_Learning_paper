# coding=utf-8
# Copyright 2020 Lucas Cassano.

"""TF2 Qtran Implementation."""

# Import all packages
from catch_prey.utils import batched_index
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda
from catch_prey import replay2
import tensorflow as tf


class Qtran(object):
  """A TensorFlow2 implementation of Qtran."""

  def __init__(
    self,
    test_observations,
    nmbr_episodes,
    max_schedule,
    nmbr_agents,
    obs_shape,
    action_spec,
    q_bound,
    num_units_q,
    num_hidden_layers_q,
    num_units_joint,
    num_hidden_layers_joint,
    num_units_v,
    num_hidden_layers_v,
    batch_size,
    discount,
    temp_max,
    temp_min,
    replay_capacity,
    min_replay_size,
    target_update_period,
    optimizer_q,
    optimizer_v,
    optimizer_joint_q,
    learn_iters,
    device='cpu:*',
    seed=None,
    save_path=''
  ):

    self._save_path = save_path
    self._device = device
    self._optimizer_q = optimizer_q
    self._optimizer_v = optimizer_v
    self._optimizer_joint_q = optimizer_joint_q
    # Hyperparameters.
    self._nmbr_agents = nmbr_agents
    self._batch_size = batch_size
    self._batch_size_tf = tf.constant(batch_size, dtype=tf.float32)
    self._num_actions = action_spec
    self._learning_updates = learn_iters
    self._target_update_period = target_update_period
    self._gamma = discount
    self._total_steps = 0
    self._eps_count = 0
    self._replay = replay2.Replay(capacity=replay_capacity, num_agents=nmbr_agents, obs_shape=obs_shape[0])
    self._min_replay_size = min_replay_size
    self._temp_min = tf.constant(temp_min, dtype=tf.float32)
    self._temp_max = tf.constant(temp_max, dtype=tf.float32)
    self._temp = tf.Variable(temp_max, dtype=tf.float32)
    self._max_schedule = max_schedule
    self._learn_iter_counter = 0

    state_shape = (nmbr_agents * 2 + 1, )

    q_out_fn = Lambda(lambda x: q_bound * tf.tanh(x))

    self._q_fact = Sequential()
    self._q_fact.add(Dense(num_units_q, activation='relu', use_bias=True, input_shape=obs_shape))
    for _ in range(num_hidden_layers_q - 1):
      self._q_fact.add(Dense(num_units_q, activation='relu', use_bias=True))
    self._q_fact.add(Dense(action_spec, activation=None, use_bias=True))

    self._q_fact_t = Sequential()
    self._q_fact_t.add(Dense(num_units_q, activation='relu', use_bias=True, input_shape=obs_shape))
    for _ in range(num_hidden_layers_q - 1):
      self._q_fact_t.add(Dense(num_units_q, activation='relu', use_bias=True))
    self._q_fact_t.add(Dense(action_spec, activation=None, use_bias=True))

    self._q_joint = Sequential()
    self._q_joint.add(Dense(num_units_joint, activation='relu', use_bias=True, input_shape=tuple([state_shape[0]+nmbr_agents*action_spec])))
    for _ in range(num_hidden_layers_joint - 1):
      self._q_joint.add(Dense(num_units_joint, activation='relu', use_bias=True))
    self._q_joint.add(Dense(1, activation=q_out_fn, use_bias=True))

    self._q_joint_t = Sequential()
    self._q_joint_t.add(Dense(num_units_joint, activation='relu', use_bias=True, input_shape=tuple([state_shape[0]+nmbr_agents*action_spec])))
    for _ in range(num_hidden_layers_joint - 1):
      self._q_joint_t.add(Dense(num_units_joint, activation='relu', use_bias=True))
    self._q_joint_t.add(Dense(1, activation=q_out_fn, use_bias=True))

    self._v = Sequential()
    self._v.add(Dense(num_units_v, activation='relu', use_bias=True, input_shape=state_shape))
    for _ in range(num_hidden_layers_v - 1):
      self._v.add(Dense(num_units_v, activation='relu', use_bias=True))
    self._v.add(Dense(1, use_bias=True))

  @tf.function
  def _learn(self, o_t, a_t, r_t, d_tp1, o_tp1):
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
      tape.watch(self._q_fact.trainable_weights + self._q_joint.trainable_weights + self._v.trainable_weights)

      distances = o_t[:, :, 0]
      angles = o_t[:, :, 3]
      time = o_t[:, 0, 0]
      s_t = tf.concat([distances, angles, tf.expand_dims(time, axis=1)], axis=1)
      distances_tp1 = o_tp1[:, :, 0]
      angles_tp1 = o_tp1[:, :, 3]
      time_tp1 = o_tp1[:, 0, 0]
      s_tp1 = tf.concat([distances_tp1, angles_tp1, tf.expand_dims(time_tp1, axis=1)], axis=1)

      # Network outputs
      v_t = self._v(s_t, training=True)
      one_hot_a_t = tf.reshape(tf.one_hot(a_t, self._num_actions, dtype=s_t.dtype),shape=(self._batch_size, self._num_actions * self._nmbr_agents))
      q_joint = self._q_joint(tf.concat([s_t,one_hot_a_t], axis=1), training=True)
      q_fact = tf.stack([self._q_fact(o_t[:, k, :], training=True) for k in range(self._nmbr_agents)], axis=1)
      opt_a_t = tf.stop_gradient(tf.argmax(q_fact, axis=-1, output_type=a_t.dtype))
      correct_acts = tf.math.equal(opt_a_t, a_t)

      q_tilde = tf.math.reduce_sum(batched_index(q_fact, a_t), axis=1)

      q_fact_tp1 = tf.stack([self._q_fact_t(o_tp1[:, k, :], training=True) for k in range(self._nmbr_agents)], axis=1)
      opt_a_tp1 = tf.stop_gradient(tf.argmax(q_fact_tp1, axis=-1, output_type=a_t.dtype))
      one_hot_a_tp1 = tf.reshape(tf.one_hot(opt_a_tp1, self._num_actions, dtype=s_tp1.dtype),shape=(self._batch_size, self._num_actions * self._nmbr_agents))
      q_joint_tp1 = self._q_joint_t(tf.concat([s_tp1, one_hot_a_tp1], axis=1), training=True)

      target = tf.stop_gradient(r_t + self._gamma * d_tp1 * q_joint_tp1)

      # Calculate errors
      delta_joint = q_joint - target
      delta_fact = q_tilde + v_t - tf.stop_gradient(q_joint)
      delta_fact_min = tf.minimum(delta_fact, 0)

      # Calculate loss
      c = tf.stop_gradient(tf.math.reduce_all(correct_acts, axis=1))
      not_c = tf.math.logical_not(c)
      c_float = tf.stop_gradient(tf.cast(c, target.dtype))
      not_c_float = tf.stop_gradient(tf.cast(not_c, target.dtype))

      loss_fact = tf.math.add(c_float * tf.square(delta_fact), not_c_float * tf.square(delta_fact_min), name='loss_fact')
      loss_joint = tf.square(delta_joint, name='loss_td')

    #Backprop
    q_fact_variables_to_train = self._q_fact.trainable_weights
    v_variables_to_train = self._v.trainable_weights
    q_joint_variables_to_train = self._q_joint.trainable_weights
    q_fact_grads = tape.gradient(loss_fact, q_fact_variables_to_train)
    v_grads = tape.gradient(loss_fact, v_variables_to_train)
    q_joint_grads = tape.gradient(loss_joint, q_joint_variables_to_train)
    self._optimizer_q.apply_gradients(list(zip(q_fact_grads, q_fact_variables_to_train)))
    self._optimizer_v.apply_gradients(list(zip(v_grads, v_variables_to_train)))
    self._optimizer_joint_q.apply_gradients(list(zip(q_joint_grads, q_joint_variables_to_train)))

  @tf.function
  def _update_target_nets(self):
    with tf.device(self._device):
      source_variables = self._q_fact.trainable_weights
      target_variables = self._q_fact_t.trainable_weights
      for (v_s, v_t) in zip(source_variables, target_variables):
        v_t.shape.assert_is_compatible_with(v_s.shape)
        v_t.assign(v_s)

      source_variables = self._q_joint.trainable_weights
      target_variables = self._q_joint_t.trainable_weights
      for (v_s, v_t) in zip(source_variables, target_variables):
        v_t.shape.assert_is_compatible_with(v_s.shape)
        v_t.assign(v_s)

  @tf.function(input_signature=[
    tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
    tf.TensorSpec(shape=( ), dtype=tf.bool),
    tf.TensorSpec(shape=(2,), dtype=tf.int32)])
  def policy(self, obs, exploit, c1_c2_games):

    @tf.function
    def exploratoy_policy(temp):
      q_values = self._q_fact(obs)
      return tf.squeeze(tf.random.categorical(tf.squeeze(q_values) / temp, 1, dtype=tf.int32))

    @tf.function
    def greedy_policy():
      q_values = self._q_fact(obs)
      return tf.squeeze(tf.argmax(q_values, axis=-1, output_type=tf.int32))

    if exploit:
      actions = greedy_policy()
    else:
      actions = exploratoy_policy(temp=tf.math.maximum(self._temp_min, self._temp_max * (1 - self._eps_count / self._max_schedule)))

    return actions

  def store_data(self, obs, actions, reward, discount, new_obs, active_games):
    """Stores new data in the replay buffer."""

    new_data = {'old_obs': obs[active_games],
                'actions': actions[active_games],
                'rewards': reward[active_games],
                'discount': discount[active_games],
                'new_obs': new_obs[active_games]}

    self._replay.add(new_data)

  def update(self):
    self._eps_count += 1
    if self._replay.size >= self._min_replay_size:
      for _l in range(self._learning_updates):
        _, minibatch = self._replay.sample(self._batch_size)
        tf_minibatch = [tf.constant(mat, dtype=tf_type) for mat, tf_type in zip(minibatch, [tf.float32, tf.int32, tf.float32, tf.float32, tf.float32])]
        self._learn(*tf_minibatch)

        if self._learn_iter_counter % self._target_update_period == 0:
          self._update_target_nets()

        self._learn_iter_counter += 1

  def load_model(self):
    """Load network weights"""
    self._q_fact.load_weights(self._save_path + 'q_fact_net.h5')
    self._q_fact_t.load_weights(self._save_path + 'q_fact_target_net.h5')
    self._q_joint.load_weights(self._save_path + 'q_joint_net.h5')
    self._q_joint_t.load_weights(self._save_path + 'q_joint_target_net.h5')
    self._v.load_weights(self._save_path + 'v_net.h5')

  def save_model(self):
    """Save network weights"""
    self._q_fact.save_weights(self._save_path + 'q_fact_net.h5')
    self._q_fact_t.save_weights(self._save_path + 'q_fact_target_net.h5')
    self._q_joint.save_weights(self._save_path + 'q_joint_net.h5')
    self._q_joint_t.save_weights(self._save_path + 'q_joint_target_net.h5')
    self._v.save_weights(self._save_path + 'v_net.h5')

