
#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic Long-Short Term Memory with Batch Normalization."""

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.python.platform import tf_logging as logging
import numpy as np


def identity_initializer(scale=1.0):
    def _initializer(shape, dtype=tf.float32):
        if len(shape) == 1:
            return tf.constant(0., dtype=dtype, shape=shape)
        elif len(shape) == 2 and shape[0] == shape[1]:
            return tf.constant(scale * np.identity(shape[0], dtype))
        elif len(shape) == 4 and shape[2] == shape[3]:
            array = np.zeros(shape, dtype=float)
            cx, cy = shape[0] / 2, shape[1] / 2
            for i in range(shape[2]):
                array[cx, cy, i, i] = 1
            return tf.constant(scale * array, dtype=dtype)
        else:
            raise
    return _initializer


def orthogonal_initializer(scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if partition_info is not None:
            ValueError("Do not know what to do with partition_info in BN_LSTMCell")
        flat_shape = (shape[0], int(np.prod(shape[1:])))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        # return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)
        return tf.constant(scale * q, dtype=dtype)

    return _initializer
"""
def batch_norm(inputs, name_scope, is_training, epsilon=1e-3, decay=0.99):
    with tf.variable_scope(name_scope):
        size = inputs.get_shape().as_list()[1]

        gamma = tf.get_variable(
            'gamma', [size], initializer=tf.constant_initializer(0.1))
        # beta = tf.get_variable('beta', [size], initializer=tf.constant_initializer(0))
        beta = tf.get_variable('beta', [size])

        pop_mean = tf.get_variable('pop_mean', [size],
                                   initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable('pop_var', [size],
                                  initializer=tf.ones_initializer(), trainable=False)
        batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean_op = tf.assign(
            pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(
            pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)

        def pop_statistics():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)

        # control flow
        return tf.cond(tf.cast(is_training,dtype=bool), batch_statistics, pop_statistics)
"""

"""
def batch_norm(x, name_scope, is_training, epsilon=1e-3, decay=0.999):
    '''Assume 2d [batch, values] tensor'''
    with tf.variable_scope(name_scope):
        return tf.contrib.layers.batch_norm(inputs=x, scale=True, epsilon=epsilon, decay=decay, updates_collections=None, is_training=is_training)
"""



def batch_norm(x, name_scope, is_training):
    '''Assume 2d [batch, values] tensor'''
    with tf.variable_scope(name_scope):
        return tf.layers.batch_normalization(inputs=x,training=is_training,fused=True)



class BatchNormBasicLSTMCell(RNNCell):
    """Batch Normalized Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full LSTMCell that follows.
    """

    def __init__(self, num_units, is_training, forget_bias=1.0, input_size=None,
                 state_is_tuple=True, reuse=None):
        """Initialize the basic LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          is_training: bool, set True when training.
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
        """
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._reuse = reuse
        self._is_training = is_training

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM) with Recurrent Batch Normalization."""
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]")

        with tf.variable_scope(scope or "batch_norm_lstm_cell", reuse=self._reuse):
            # Parameters of gates are concatenated into one multiply for
            # efficiency.
            if self._state_is_tuple:
                c_prev, h_prev = state
            else:
                c_prev, h_prev = tf.split(
                    value=state, num_or_size_splits=2, axis=1)

            W_xh = tf.get_variable('W_xh', shape=[input_size, 4 * self._num_units],
                                   initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh', shape=[self._num_units, 4 * self._num_units],
                                   initializer=orthogonal_initializer())
            bias = tf.get_variable('b', [4 * self._num_units])

            xh = tf.matmul(inputs, W_xh)
            hh = tf.matmul(h_prev, W_hh)

            bn_xh = batch_norm(xh, 'xh', self._is_training)
            bn_hh = batch_norm(hh, 'hh', self._is_training)

            # i = input_gate, g = new_input, f = forget_gate, o = output_gate
            # lstm_matrix = tf.contrib.rnn._linear([inputs, h_prev], 4 * self._num_units, True)
            lstm_matrix = tf.nn.bias_add(tf.add(bn_xh, bn_hh), bias)
            i, g, f, o = tf.split(
                value=lstm_matrix, num_or_size_splits=4, axis=1)

            c = (c_prev * tf.sigmoid(f + self._forget_bias) +
                 tf.sigmoid(i) * tf.tanh(g))

            bn_c = batch_norm(c, 'bn_c', self._is_training)

            h = tf.tanh(bn_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(c, h)
            else:
                new_state = tf.concat(values=[c, h], axis=1)
            return h, new_state




