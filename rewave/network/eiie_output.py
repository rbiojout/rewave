from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from math import sqrt
import numpy as np
import tensorflow as tf

from tensorforce import TensorForceError, util
import tensorforce.core.networks

from tensorforce.core.networks import Layer, Conv2d, Nonlinearity
from tensorforce.core.networks import LayeredNetwork, layers, Network, network


"""
size=20
window_length = 50


network_spec_raw_test = [ 
    [
        dict(type='input', inputs=['history']),
        dict(type='conv2d', size=3, bias=True, stride=1,window=(1,3),padding='VALID',l1_regularization=.00,l2_regularization=.000),
        dict(type='conv2d', size=size, bias=True, stride=(1,window_length-2-1),window=(1,window_length-2-1),padding='VALID',l1_regularization=.00,l2_regularization=.000),
        dict(type='output', output='actions'),   
    ],
    [
        dict(type='input', inputs=['actions']),                  
        dict(type='eiieoutput',l1_regularization=.00,l2_regularization=.000),
    ]    
]   



network_spec_raw_test = [ 
    [
        dict(type='input', inputs=['mp','env']),
        dict(type='dense', size=None, activation={'name':'swish' ,'beta': 'learn'}, skip=True,l1_regularization=.00,l2_regularization=.000),
        dict(type='dense', size=15, activation={'name':'swish' ,'beta': 1.9}, skip=False,l1_regularization=.00,l2_regularization=.000)
    ],
    [
        dict(type='input', inputs=['mp','*']),                  
        dict(type='dense', size=None, activation='swish', skip=True,l1_regularization=.00,l2_regularization=.000),
        dict(type='dense', size=10, activation='swish', skip=False,l1_regularization=.00,l2_regularization=.000),
        dict(type='dense', size=10, activation='swish', skip=False,l1_regularization=.00,l2_regularization=.000),   
    ]    
]   
"""

"""
ComplexNetwork basic implementation:
network_spec_CL_selu_cp = [ 
    [
        dict(type='input', inputs=['state']),
        dict(type='dense', size=50, activation='selu', skip=False,l1_regularization=.00,l2_regularization=.0002),
        dict(type='dense', size=25, activation='selu', skip=False,l1_regularization=.00,l2_regularization=.0002),
        dict(type='dueling', size=3, activation='none', output=('expectation1','advantage1','mean_advantage2'),
            summary_labels=['activations','expectation','advantage','mean_advantage']),               
        dict(type='output', output='skip'),
        dict(type='dense', size=3, activation='selu', skip=False,l1_regularization=.00,l2_regularization=.0002),
        dict(type='input', inputs=['*','skip']),            
        dict(type='dense', size=3, activation='selu', skip=False,l1_regularization=.00,l2_regularization=.0002),
        dict(type='dueling', size=3, activation='none', output=('expectation2','advantage2','mean_advantage2'),
            summary_labels=['activations','expectation','advantage','mean_advantage']),
        dict(type='output', output='actions'),   
    ],
    [
        dict(type='input', inputs=['*','advantage1']), 
        #dict(type='input', inputs=['actions','advantage']),    # Same as above                    
        dict(type='dense', size=None, activation='selu', skip=True,l1_regularization=.00,l2_regularization=.0002),
        dict(type='dense', size=None, activation='selu', skip=True,l1_regularization=.00,l2_regularization=.0002),
        dict(type='dueling', size=3, activation='none', output=('expectation3','advantage3','mean_advantage3'),
            summary_labels=['activations','expectation','advantage','mean_advantage']),            
    ]    
] 
"""



"""
            elif layer["type"] == "EIIE_Output":
                width = network.get_shape()[2]
                # default for conv2d : weights_init='truncated_normal',regularizer=None, weight_decay=0.001
                network = tflearn.layers.conv_2d(network, nb_filter=1,
                                                 filter_size=[1, width],
                                                 strides=1,
                                                 padding="valid",
                                                 activation='linear', bias=True,
                                                 # weights_init='uniform_scaling',
                                                 weights_init='uniform',
                                                 bias_init='zeros',
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"],
                                                 trainable=True, restore=True, reuse=False, scope=None,
                                                 name="EIIE_Output")
                network = network[:, :, 0, 0]
                btc_bias = tf.ones((self.input_num, 1))
                network = tf.concat([btc_bias, network], 1)
                network = tflearn.layers.core.activation(network, activation="softmax")
            elif layer["type"] == "Output_WithW":
                network = tflearn.flatten(network)
                network = tf.concat([network,self.previous_w], axis=1)
                # default for fully_connected : regularizer=None, weight_decay=0.001
                network = tflearn.fully_connected(network, n_units=self._rows+1,
                                                  activation="softmax",
                                                  bias=True,
                                                  weights_init='truncated_normal', bias_init='zeros',
                                                  regularizer=layer["regularizer"],
                                                  weight_decay=layer["weight_decay"],
                                                  trainable=True,
                                                  restore=True, reuse=False, scope=None,
                                                  name="Output_WithW"                                                  )
            elif layer["type"] == "EIIE_Output_WithW":
                width = network.get_shape()[2]
                height = network.get_shape()[1]
                features = network.get_shape()[3]
                network = tf.reshape(network, [self.input_num, int(height), 1, int(width*features)])
                w = tf.reshape(self.previous_w, [-1, int(height), 1, 1])
                network = tf.concat([network, w], axis=3)
                # default for conv2d : regularizer=None, weight_decay=0.001
                network = tflearn.layers.conv_2d(network, nb_filter=1, filter_size=[1, 1],
                                                 strides=1,
                                                 padding="valid",
                                                 activation='linear', bias=True,
                                                 weights_init='uniform_scaling',
                                                 bias_init='zeros',
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"],
                                                 trainable=True, restore=True, reuse=False, scope=None,
                                                 name="EIIE_Output_WithW")
                network = network[:, :, 0, 0]
                #btc_bias = tf.zeros((self.input_num, 1))
                btc_bias = tf.get_variable("btc_bias", [1, 1], dtype=tf.float32,
                                            initializer = tf.zeros_initializer)
                btc_bias = tf.tile(btc_bias, [self.input_num, 1])
                network = tf.concat([btc_bias, network], 1)
                self.voting = network
                network = tflearn.layers.core.activation(network, activation="softmax")
"""




"""
EIIE Output layer
use of two entries:
- * : the precedent treatments 4-D Tensor [batch, height, width, in_channels]
    should be linked to [batch, assets, time, features]
- last_w : the last weights coming from environment
"""


"""
            width = network.get_shape()[2]
            # default for conv2d : weights_init='truncated_normal',regularizer=None, weight_decay=0.001
            network = tflearn.layers.conv_2d(network, nb_filter=1,
                                             filter_size=[1, width],
                                             strides=1,
                                             padding="valid",
                                             activation='linear', bias=True,
                                             # weights_init='uniform_scaling',
                                             weights_init='uniform',
                                             bias_init='zeros',
                                             regularizer=layer["regularizer"],
                                             weight_decay=layer["weight_decay"],
                                             trainable=True, restore=True, reuse=False, scope=None,
                                             name="EIIE_Output")
            network = network[:, :, 0, 0]
            btc_bias = tf.ones((self.input_num, 1))
            network = tf.concat([btc_bias, network], 1)
            network = tflearn.layers.core.activation(network, activation="softmax")
"""



class EIIE_OutPut(Layer):
    """
    EIIE Output layer
    based on 2-dimensional convolutional layer.
    use of two entries:
    - * : the precedent treatments
    - last_w : the last weights coming from environment
    """

    def __init__(
            self,
            l2_regularization=0.0,
            l1_regularization=0.0,
            scope='eieeoutput',
            summary_labels=()
    ):
        """
        2D convolutional layer.

        Args:
            size: Number of filters set to 1
            window: Convolution window size, either an integer or pair of integers. calculated
            stride: Convolution stride, either an integer or pair of integers.
            padding: Convolution padding, one of 'VALID' or 'SAME'
            bias: If true, a bias is added
            activation: Type of nonlinearity, or dict with name & arguments
            l2_regularization: L2 regularization weight
            l1_regularization: L1 regularization weight
        """
        self.size = 1
        self.stride = 1
        self.padding = 'VALID'
        self.bias = True
        activation = 'relu'
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.nonlinearity = Nonlinearity(name=activation, summary_labels=summary_labels)
        super(EIIE_OutPut, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        if util.rank(x) != 4:
            raise TensorForceError('Invalid input rank for conv2d layer: {}, must be 4'.format(util.rank(x)))

        self.window = (1, x.shape[2])
        filters_shape = self.window + (x.shape[3].value, self.size)
        stddev = min(0.1, sqrt(2.0 / self.size))
        filters_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
        self.filters = tf.get_variable(name='W', shape=filters_shape, dtype=tf.float32, initializer=filters_init)
        stride_h, stride_w = self.stride if type(self.stride) is tuple else (self.stride, self.stride)
        x = tf.nn.conv2d(input=x, filter=self.filters, strides=(1, stride_h, stride_w, 1), padding=self.padding)

        if self.bias:
            bias_shape = (self.size,)
            bias_init = tf.zeros_initializer(dtype=tf.float32)
            self.bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=bias_init)
            x = tf.nn.bias_add(value=x, bias=self.bias)

        x = self.nonlinearity.apply(x=x, update=update)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x

    def tf_regularization_loss(self):
        regularization_loss = super(EIIE_OutPut, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.filters))
            if self.bias is not None:
                losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.bias))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.filters)))
            if self.bias is not None:
                losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.bias)))

        regularization_loss = self.nonlinearity.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        layer_variables = super(EIIE_OutPut, self).get_variables(include_non_trainable=include_non_trainable)
        nonlinearity_variables = self.nonlinearity.get_variables(include_non_trainable=include_non_trainable)

        return layer_variables + nonlinearity_variables

    def get_summaries(self):
        layer_summaries = super(EIIE_OutPut, self).get_summaries()
        nonlinearity_summaries = self.nonlinearity.get_summaries()

        return layer_summaries + nonlinearity_summaries

# Add our custom layer
layers['eiieoutput'] = EIIE_OutPut