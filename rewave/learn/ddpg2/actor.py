from keras import backend as K

import tensorflow as tf

from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers import Layer, Input, Dense, BatchNormalization, Activation, Lambda, Flatten, Dropout, ELU
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.initializers import RandomUniform, Constant
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.constraints import min_max_norm

from tensorflow.python.ops import math_ops
from tensorflow.python.keras._impl.keras import activations

from tensorflow.python.keras.optimizers import Adam

import tensorflow as tf

DEBUG = True


class Actor(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, state_input, root_net, action_dim, lr=0.001):
        """

        :param input_state: the input state layer for the actor
        :param root_net: the network shared between actor and critic
        """
        self.state_input = state_input
        self.root_net = root_net
        self.action_dim = action_dim
        self.lr = lr

        self.use_batch_norm =False

        self.net = self.create_actor_model(self.state_input, self.root_net, self.action_dim, self.lr)

    def create_actor_model(self, state_input, root_net, action_dim, lr=0.001, dropout=0.3):
        """
        net = Dense(64, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01), activation="relu")(root_net)
        """
        net = Dense(40, activation="relu")(root_net)
        #net = Dropout(dropout)(net)
        if self.use_batch_norm:
            net = BatchNormalization()(net)
        """
        net = Dense(64, input_dim=64,
                    kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01), activation="relu")(net)
        """
        net = Flatten()(net)
        net = Dense(action_dim * 2, activation="sigmoid")(net)
        #net = Dropout(dropout)(net)
        if self.use_batch_norm:
            net = BatchNormalization()(net)


        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        random_initializer = RandomUniform(minval=0.01, maxval=0.1, seed=None)

        # add input
        # @TODO add the previous action

        actor_out = Dense(action_dim,
                          #kernel_initializer=random_initializer,
                          #kernel_initializer="glorot_uniform", # for softmax
                          #kernel_initializer="he_uniform", # for relu
                          activation="softmax",
                          # activation="relu",
                          kernel_initializer= Constant(1.0/action_dim),
                          name="actor_out")(net)

        # actor_out = CustomActivation()(actor_out)
        #actor_out = Lambda(lambda x: tf.sigmoid(x) /  (1e-5+ tf.norm(tf.sigmoid(x), axis=0, ord=1, keep_dims=True)))(actor_out)

        actor_model = Model(inputs=state_input, outputs=actor_out)
        if DEBUG:
            print("ACTOR MODEL :", actor_model.summary())


        return actor_model

    def references(self):
        return self.state_input, self.net
