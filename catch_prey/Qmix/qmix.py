# coding=utf-8
# Copyright 2020 Lucas Cassano.

"""TF2 Qmix Implementation."""

# Import all packages
from catch_prey.utils import QmixNet
from catch_prey import replay2
import tensorflow as tf


class Qmix(object):
  """A TensorFlow2 implementation of Qmix."""

  def __init__(
    self,
    test_observations,
    nmbr_episodes,
    max_schedule,
    nmbr_agents,
    obs_shape,
    action_spec,
    num_units_q_net,
    num_hidden_layers_q_net,
    num_units_hyper,
    num_units_mixer,
    batch_size,
    discount,
    temp_max,
    temp_min,
    replay_capacity,
    min_replay_size,
    target_update_period,
    optimizer_q,
    learn_iters,
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
    self._qmix_net = QmixNet(nmbr_agents=nmbr_agents,
                             obs_shape=obs_shape,
                             num_hidden_layers_q_net=num_hidden_layers_q_net,
                             num_units_q_net=num_units_q_net,
                             action_spec=action_spec,
                             num_units_hyper=num_units_hyper,
                             num_units_mixer=num_units_mixer,
                             state_shape=state_shape)

    self._qmix_target_net = QmixNet(nmbr_agents=nmbr_agents,
                                    obs_shape=obs_shape,
                                    num_hidden_layers_q_net=num_hidden_layers_q_net,
                                    num_units_q_net=num_units_q_net,
                                    action_spec=action_spec,
                                    num_units_hyper=num_units_hyper,
                                    num_units_mixer=num_units_mixer,
                                    state_shape=state_shape)

  # @tf.function(input_signature=[
  #       tf.TensorSpec(shape=[None, 3, 7], dtype=tf.float32),
  #       tf.TensorSpec(shape=[None, 3], dtype=tf.int32),
  #       tf.TensorSpec(shape=[None, ], dtype=tf.float32),
  #       tf.TensorSpec(shape=[None, ], dtype=tf.float32),
  #       tf.TensorSpec(shape=[None, 3, 7], dtype=tf.float32)])
  @tf.function
  def _learn(self, o_t, a_t, r_t, d_tp1, o_tp1):
    #with tf.device(self._device):
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
      tape.watch(self._qmix_net.trainable_weights)

      distances = o_t[:, :, 0]
      angles = o_t[:, :, 3]
      time = o_t[:, 0, 0]
      s_t = tf.concat([distances, angles, tf.expand_dims(time, axis=1)], axis=1)

      _, joint_q = self._qmix_net(o_t, a_t, s_t)

      q_tp1 = tf.stack([tf.reduce_max(self._qmix_target_net.eval_q(o_tp1[:, k, :]), axis=-1) for k in range(self._nmbr_agents)], axis=1)
      distances = o_tp1[:, :, 0]
      angles = o_tp1[:, :, 3]
      time = o_tp1[:, 0, 0]
      s_tp1 = tf.concat([distances, angles, tf.expand_dims(time, axis=1)], axis=1)
      q_tp1 = self._qmix_target_net.mix(q_tp1, s_tp1)
      q_target_value = tf.stop_gradient(r_t + self._gamma * d_tp1 * q_tp1)

      loss = 0.5 * tf.square(joint_q - q_target_value, name='loss')

    variables_to_train = self._qmix_net.trainable_weights
    grads = tape.gradient(loss, variables_to_train)
    self._optimizer_q.apply_gradients(list(zip(grads, variables_to_train)))

  @tf.function
  def _update_target_nets(self):
    with tf.device(self._device):
      source_variables = self._qmix_net.trainable_weights
      target_variables = self._qmix_target_net.trainable_weights
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
      q_values = self._qmix_target_net.eval_q(obs)
      return tf.squeeze(tf.random.categorical(tf.squeeze(q_values) / temp, 1, dtype=tf.int32))

    @tf.function
    def greedy_policy():
      q_values = self._qmix_target_net.eval_q(obs)
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
    self._qmix_net.load_weights(self._save_path + 'qmix_net.h5')
    self._qmix_target_net.load_weights(self._save_path + 'qmix_target_net.h5')

  def save_model(self):
    """Save network weights"""
    self._qmix_net.save_weights(self._save_path + 'qmix_net.h5')
    self._qmix_target_net.save_weights(self._save_path + 'qmix_target_net.h5')

