#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for constructing RNN Cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy
import tensorflow as tf

from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers

# import nabu.neuralnetworks.components.new_layer as layers

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables  # pylint: disable=unused-import
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import layers

###################################################################################

import numpy as np

###################################################################################

class LN_LSTMCell(rnn_cell_impl.RNNCell):

		"""
		LSTM unit with layer normalization 
		Layer normalization implementation is based on:
		https://arxiv.org/abs/1607.06450.
	
		"""

		def __init__(
				self,
				num_units,
				forget_bias=1.0,
				input_size=None,
				reuse=None,
				use_layer_norm=True,
				use_recurrent_dropout=False, dropout_keep_prob=0.90,
				training= True,
				):
				"""
				Initializes the basic LSTM cell.
				Args:
					num_units: int, The number of units in the LSTM cell.
					forget_bias: float, The bias added to forget gates (see above).
					input_size: Deprecated and unused.
					
				"""

				super(LN_LSTMCell, self).__init__(_reuse=reuse)

				if input_size is not None:
						logging.warn('%s: The input_size parameter is deprecated.',
												 self)

				self._num_units = num_units
				self._forget_bias = forget_bias
				self._reuse = reuse
				self.use_recurrent_dropout = use_recurrent_dropout
				self.dropout_keep_prob = dropout_keep_prob
				self.use_layer_norm = use_layer_norm

		@property
		def state_size(self):
				return rnn_cell_impl.LSTMStateTuple(self._num_units,
								self._num_units)

		@property
		def output_size(self):
				return self._num_units

		def call(self, x, state):
			with tf.variable_scope("ln"):
				h, c = state

				h_size = self._num_units
	  
				batch_size = x.get_shape().as_list()[0]
				x_size = x.get_shape().as_list()[1]
				  
				w_init= None # uniform

				h_init=lstm_ortho_initializer()

				W_xh = tf.get_variable('W_xh',
					[x_size, 4 * self._num_units], initializer=w_init)

				W_hh = tf.get_variable('W_hh_i',
					[self._num_units, 4*self._num_units], initializer=h_init)

				bias = tf.get_variable('bias',
					[4 * self._num_units], initializer=tf.constant_initializer(0.0))

				
				xh = tf.matmul(x,W_xh)
				hh = tf.matmul(h,W_hh)
				ln_xh = raw_layer_norm(xh)
				ln_hh = raw_layer_norm(hh)
				concat = ln_xh + ln_hh + bias 
				#concat = xh + hh + bias
				i, j, f, o = tf.split(concat, 4, 1)
				g = tf.tanh(j) 
				new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i)*g
				new_h = tf.tanh(layer_norm(new_c,self._num_units,scope='ln_c')) * tf.sigmoid(o)
				
				return new_h, tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
		

		

		


###################################### 
#custom code for layer normalization #
######################################
def lstm_ortho_initializer(scale=1.0):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
	size_x = shape[0]
	size_h = shape[1]//4 # assumes lstm.
	t = np.zeros(shape)
	t[:, :size_h] = orthogonal([size_x, size_h])*scale
	t[:, size_h:size_h*2] = orthogonal([size_x, size_h])*scale
	t[:, size_h*2:size_h*3] = orthogonal([size_x, size_h])*scale
	t[:, size_h*3:] = orthogonal([size_x, size_h])*scale
	return tf.constant(t, dtype)
  return _initializer

def orthogonal(shape):
	flat_shape = (shape[0], np.prod(shape[1:]))
	a = np.random.normal(0.0, 1.0, flat_shape)
	u, _, v = np.linalg.svd(a, full_matrices=False)
	q = u if u.shape == flat_shape else v
	return q.reshape(shape)

def orthogonal_initializer(scale=1.0):
	def _initializer(shape, dtype=tf.float32, partition_info=None):
		return tf.constant(orthogonal(shape) * scale, dtype)
	return _initializer






def layer_norm(x, num_units, scope="layer_norm", reuse=False, gamma_start=1.0, epsilon = 1e-5, use_bias=True):
  axes = [1]
  mean = tf.reduce_mean(x, axes, keep_dims=True)
  x_shifted = x-mean
  var = tf.reduce_mean(tf.square(x_shifted), axes, keep_dims=True)
  inv_std = tf.rsqrt(var + epsilon)
  with tf.variable_scope(scope):
    if reuse == True:
      tf.get_variable_scope().reuse_variables()
    gamma = tf.get_variable('ln_gamma', [num_units], initializer=tf.constant_initializer(gamma_start))
    if use_bias:
      beta = tf.get_variable('ln_beta', [num_units], initializer=tf.constant_initializer(0.0))
  output = gamma*(x_shifted)*inv_std
  if use_bias:
    output = output + beta
  return output

def raw_layer_norm(x, epsilon=1e-3):
	  axes = [1]
	  mean = tf.reduce_mean(x, axes, keep_dims=True)
	  std = tf.sqrt(
	      tf.reduce_mean(tf.square(x - mean), axes, keep_dims=True) + epsilon)
	  output = (x - mean) / (std)
	  return output
	

