
# Import all packages
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape


class QmixNet(tf.keras.Model):
  def __init__(self, dims, input_shape, name='Qmix', **kwargs):
    super(QmixNet, self).__init__(name=name, **kwargs)

    q_init = tf.zeros_initializer()
    self._q1 = tf.Variable(initial_value=q_init(shape=dims[0], dtype='float32'), trainable=True)
    self._q2 = tf.Variable(initial_value=q_init(shape=dims[1], dtype='float32'), trainable=True)

    num_units_hyper = 5
    num_units = 10

    self._b0_net = Sequential()
    self._b0_net.add(Dense(num_units, input_shape=input_shape))

    self._w0_net = Sequential()
    self._w0_net.add(Dense(num_units_hyper, activation='relu', input_shape=input_shape))
    self._w0_net.add(Dense(num_units * 2, activation=tf.math.abs))
    self._w0_net.add(Reshape(target_shape=(2, num_units)))

    self._b1_net = Sequential()
    self._b1_net.add(Dense(num_units_hyper, activation='relu', input_shape=input_shape))
    self._b1_net.add(Dense(1, input_shape=input_shape))

    self._w1_net = Sequential()
    self._w1_net.add(Dense(num_units_hyper, activation='relu', input_shape=input_shape))
    self._w1_net.add(Dense(num_units, activation=tf.math.abs))
    self._w1_net.add(Reshape(target_shape=(num_units, 1)))

  @tf.function
  def call(self, obs1, obs2, actions):

    q1 = self._q1[obs1[0], obs1[1]]
    q2 = self._q2[obs2[0], obs2[1]]
    q1 = q1[actions[0]]
    q2 = q2[actions[1]]

    state = tf.expand_dims(obs2, axis=0)
    b_0 = self._b0_net(state)
    w_0 = self._w0_net(state)
    b_1 = self._b1_net(state)
    w_1 = self._w1_net(state)

    x = tf.expand_dims(tf.stack([q1, q2]), axis=0)
    x = tf.matmul(x, w_0) + b_0
    x = tf.nn.elu(x)
    output = tf.matmul(x, w_1) + b_1

    return q1, q2, output

  @tf.function
  def mix(self, q1, q2, state):

    state = tf.expand_dims(state, axis=0)
    b_0 = self._b0_net(state)
    w_0 = self._w0_net(state)
    b_1 = self._b1_net(state)
    w_1 = self._w1_net(state)

    x = tf.expand_dims(tf.stack([q1, q2]), axis=0)
    x = tf.matmul(x, w_0) + b_0
    x = tf.nn.elu(x)
    output = tf.matmul(x, w_1) + b_1

    return output

  @tf.function
  def eval_q(self, obs1, obs2):
    q1 = self._q1[obs1[0], obs1[1]]
    q2 = self._q2[obs2[0], obs2[1]]

    return q1, q2

  @tf.function
  def eval_q2(self, obs):
    return self._q2[obs[0], obs[1]]


class Qmix(object):
  """TF2 Qmix."""

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
      epsilon,
      discount,
      seed=None,
  ):

    dims = [np.concatenate([obs_size[0], [n_actions_per_agent[0]]]),
            np.concatenate([obs_size[1], [n_actions_per_agent[1]]])]

    self._optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate/100)
    self._q_mixer = QmixNet(dims, input_shape=(2, ))
    self._q_mixer_target = QmixNet(dims, input_shape=(2,))
    self._update_target_period = 10

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
    self._steps_count = 0
    self._rng = np.random.RandomState(seed)
    tf.random.set_seed(seed)
    self._plot_points = 1e3
    self._plot_period = int(np.max([num_steps//self._plot_points, 1]))
    self._q_history = np.empty([grid_length, num_steps//self._plot_period, n_actions_per_agent[1]])

  def policy(self, observations, exploit):
    """Select actions according to epsilon-greedy policy."""

    actions = np.empty([self._nmbr_types], dtype=int)

    q1, q2 = self._q_mixer.eval_q(observations[0], observations[1])
    q = [q1, q2]

    for tau in range(self._nmbr_types):
      if not exploit and self._rng.rand() < np.maximum(0.05, self._epsilon * (1 - self._steps_count/5e4)):
        actions[tau] = self._rng.randint(self._n_actions_per_agent[tau])
      else:
        actions[tau] = self._rng.choice(np.flatnonzero(q[tau] == np.max(q[tau])))

    return actions

  # @tf.function(input_signature=[
  #   tf.TensorSpec(shape=(2, 2), dtype=tf.int32),
  #   tf.TensorSpec(shape=(2, ), dtype=tf.int32),
  #   tf.TensorSpec(shape=(1, ), dtype=tf.float32),
  #   tf.TensorSpec(shape=(2, 2), dtype=tf.int32),
  #   tf.TensorSpec(shape=(1, ), dtype=tf.int32)])
  @tf.function
  def learn(self, o1, o2, a, r, o_n1, o_n2, last):
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
      tape.watch(self._q_mixer.trainable_weights)

      q1, q2, q_out = self._q_mixer(o1, o2, a, training=True)

      q1_n, q2_n = self._q_mixer_target.eval_q(o_n1, o_n2)
      q_next = self._q_mixer_target.mix(tf.math.reduce_max(q1_n), tf.math.reduce_max(q2_n), o_n2)
      g = self._discount * float(1 - last)
      target = tf.stop_gradient(r + g * q_next)
      loss = 0.5 * tf.square(q_out - target, name='loss')

    mixer_grads = tape.gradient(loss, self._q_mixer.trainable_weights)
    self._optimizer.apply_gradients(list(zip(mixer_grads, self._q_mixer.trainable_weights)))

  def update(self, observations, actions, reward, new_observations, last):
    if np.mod(self._steps_count, self._plot_period) == 0:  # End of episode
      for slot in range(self._grid_length):
        obs = [slot, 1]
        q2 = self._q_mixer.eval_q2(obs)
        self._q_history[slot][self._steps_count // self._plot_period] = q2

    if np.mod(self._steps_count, self._update_target_period) == 0:  # Update target
      source_variables = self._q_mixer.trainable_weights
      target_variables = self._q_mixer_target.trainable_weights
      for (v_s, v_t) in zip(source_variables, target_variables):
        v_t.shape.assert_is_compatible_with(v_s.shape)
        v_t.assign(v_s)

    self.learn(tf.constant(observations[0], dtype=tf.int32),
               tf.constant(observations[1], dtype=tf.int32),
               tf.constant(actions, dtype=tf.int32),
               tf.constant(reward, dtype=tf.float32),
               tf.constant(new_observations[0], dtype=tf.int32),
               tf.constant(new_observations[1], dtype=tf.int32),
               tf.constant(last, dtype=tf.int32))

    self._steps_count += 1

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
    fig.savefig(fname=save_path+'qmix_0', bbox_inches='tight')

    fig2, ax2 = plt.subplots()
    plt.xlabel('Epochs', fontsize=25)
    plt.ylabel('Q function', fontsize=25)
    ax2.plot(np.arange(start=0, stop=self._q_history.shape[1] * self._plot_period, step=self._plot_period),
            self._q_history[1], linewidth=2)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.legend(['Stay', 'Left', 'Right'], fontsize=15)
    plt.grid()
    fig2.savefig(fname=save_path+'qmix_1', bbox_inches='tight')

    fig3, ax3 = plt.subplots()
    plt.xlabel('Epochs', fontsize=25)
    plt.ylabel('Q function', fontsize=25)
    ax3.plot(np.arange(start=0, stop=self._q_history.shape[1] * self._plot_period, step=self._plot_period),
            self._q_history[2], linewidth=2)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.legend(['Stay', 'Left', 'Right'], fontsize=15)
    plt.grid()
    fig3.savefig(fname=save_path+'qmix_2', bbox_inches='tight')

    fig4, ax4 = plt.subplots()
    plt.xlabel('Epochs', fontsize=25)
    plt.ylabel('Q function', fontsize=25)
    ax4.plot(np.arange(start=0, stop=self._q_history.shape[1] * self._plot_period, step=self._plot_period),
            self._q_history[3], linewidth=2)
    ax4.tick_params(axis='both', which='major', labelsize=15)
    ax4.legend(['Stay', 'Left', 'Right'], fontsize=15)
    plt.grid()
    fig4.savefig(fname=save_path+'qmix_3', bbox_inches='tight')

    plt.close('all')
