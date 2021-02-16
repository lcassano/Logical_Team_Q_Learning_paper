"""TF2 simple Qmix Implementation for matrix game."""

# Import all packages
import tensorflow as tf


class QmixNet(tf.keras.Model):
  def __init__(self, matrix_dims, name='Qmix', **kwargs):
    super(QmixNet, self).__init__(name=name, **kwargs)

    q_init = tf.zeros_initializer()
    self.q_1 = tf.Variable(initial_value=q_init(shape=(matrix_dims[0],), dtype='float32'), trainable=True)
    self.q_2 = tf.Variable(initial_value=q_init(shape=(matrix_dims[1],), dtype='float32'), trainable=True)

    nmbr_units = 5

    b_init = tf.zeros_initializer()
    self.b_0 = tf.Variable(initial_value=b_init(shape=(nmbr_units,), dtype='float32'), trainable=True)
    self.b_1 = tf.Variable(initial_value=b_init(shape=(1,), dtype='float32'), trainable=True)
    w_init = tf.random_normal_initializer()
    self.w_0 = tf.Variable(initial_value=w_init(shape=(2, nmbr_units), dtype='float32'), trainable=True)
    self.w_1 = tf.Variable(initial_value=w_init(shape=(nmbr_units, 1), dtype='float32'), trainable=True)

  @tf.function
  def call(self, actions):
    x = tf.expand_dims(tf.stack([self.q_1[actions[0]], self.q_2[actions[1]]]), axis=0)
    x = tf.matmul(x, tf.math.exp(self.w_0)) + self.b_0
    x = tf.nn.elu(x)
    output = tf.matmul(x, tf.math.exp(self.w_1)) + self.b_1
    return self.q_1[actions[0]], self.q_2[actions[1]], output


class Qmix(object):
  """Qmix for matrix game."""

  def __init__(self, matrix_dims, step_size):

    self._optimizer = tf.keras.optimizers.SGD(learning_rate=step_size)
    self._q_mixer = QmixNet(matrix_dims)

  @tf.function
  def learn(self, actions, r):
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
      tape.watch(self._q_mixer.trainable_weights)
      q1, q2, q_out = self._q_mixer(actions, training=True)
      loss = 0.5 * tf.square(q_out - r, name='loss')

    grads = tape.gradient(loss, self._q_mixer.trainable_weights)
    self._optimizer.apply_gradients(list(zip(grads, self._q_mixer.trainable_weights)))

    return q1, q2, q_out

  @tf.function
  def obtain_q(self, actions):
    """Obtain q's."""
    return self._q_mixer(actions, training=False)
