import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape


def batched_index(values, indices):

  one_hot_indices = tf.one_hot(indices, tf.shape(values)[-1], dtype=values.dtype)
  return tf.reduce_sum(values * one_hot_indices, axis=-1)


class MyNNs(tf.keras.Model):
    def __init__(self,
                 disentangle,
                 input_shape,
                 num_hidden_layers,
                 num_units,
                 actions,
                 bias_init,
                 output_nonlinearity_fn=None,
                 name='MyNNs',
                 **kwargs
                 ):
      super(MyNNs, self).__init__(name=name, **kwargs)
      self._disentangle = disentangle

      if self._disentangle:
        self._heads = actions
        self._network = [Sequential() for _ in range(actions)]
        for a in range(actions):
          self._network[a].add(Dense(num_units, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer=bias_init, input_shape=input_shape))
          for _ in range(num_hidden_layers - 1):
            self._network[a].add(Dense(num_units, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer=bias_init))
          self._network[a].add(Dense(1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer=bias_init))
          if output_nonlinearity_fn is not None:
            self._network[a].add(output_nonlinearity_fn)
      else:
        self._network = Sequential()
        self._network.add(Dense(num_units, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer=bias_init, input_shape=input_shape))
        for _ in range(num_hidden_layers - 1):
          self._network.add(Dense(num_units, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer=bias_init))
        self._network.add(Dense(actions, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer=bias_init))
        if output_nonlinearity_fn is not None:
          self._network.add(output_nonlinearity_fn)

    @tf.function
    def call(self, inputs):
      if self._disentangle:
        return tf.concat([self._network[a](inputs) for a in range(5)], axis=1)
      else:
        return self._network(inputs)


class QmixNet(tf.keras.Model):
  def __init__(self,
               nmbr_agents,
               obs_shape,
               num_hidden_layers_q_net,
               num_units_q_net,
               action_spec,
               num_units_hyper,
               num_units_mixer,
               state_shape,
               name='Qmix', **kwargs):
    super(QmixNet, self).__init__(name=name, **kwargs)

    q_out_fn = None
    self._nmbr_agents = nmbr_agents
    self._q = MyNNs(True, obs_shape, num_hidden_layers_q_net, num_units_q_net, action_spec, 'zeros', q_out_fn)

    self._b0_net = Sequential()
    self._b0_net.add(Dense(num_units_mixer, activation=None, input_shape=state_shape))

    self._w0_net = Sequential()
    self._w0_net.add(Dense(num_units_hyper, activation='relu', input_shape=state_shape))
    self._w0_net.add(Dense(num_units_mixer * nmbr_agents, activation=tf.math.abs))
    self._w0_net.add(Reshape(target_shape=(nmbr_agents, num_units_mixer)))

    self._b1_net = Sequential()
    self._b1_net.add(Dense(num_units_hyper, activation='relu', input_shape=state_shape))
    self._b1_net.add(Dense(1, activation=None, input_shape=state_shape))

    self._w1_net = Sequential()
    self._w1_net.add(Dense(num_units_hyper, activation='relu', input_shape=state_shape))
    self._w1_net.add(Dense(num_units_mixer, activation=tf.math.abs))
    self._w1_net.add(Reshape(target_shape=(num_units_mixer, 1)))

  @tf.function
  def call(self, obs, actions, state):
    q = tf.stack([batched_index(self._q(obs[:, k, :], training=True), actions[:, k]) for k in range(self._nmbr_agents)], axis=1)

    b_0 = self._b0_net(state)
    w_0 = self._w0_net(state)
    b_1 = self._b1_net(state)
    w_1 = self._w1_net(state)

    x = tf.matmul(q, w_0) + b_0
    x = tf.nn.elu(x)
    output = tf.matmul(x, w_1) + b_1

    return q, output

  @tf.function
  def mix(self, q, state):

    b_0 = self._b0_net(state)
    w_0 = self._w0_net(state)
    b_1 = self._b1_net(state)
    w_1 = self._w1_net(state)

    x = tf.matmul(q, w_0) + b_0
    x = tf.nn.elu(x)
    output = tf.matmul(x, w_1) + b_1

    return output

  @tf.function(input_signature=[tf.TensorSpec(shape=(None, 7), dtype=tf.float32)])
  def eval_q(self, obs):
    return self._q(obs)
