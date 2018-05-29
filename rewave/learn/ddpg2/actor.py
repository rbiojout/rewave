from keras import backend as K

import tensorflow as tf

from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers import Dense, BatchNormalization, Activation, Lambda
from keras import regularizers

from tensorflow.python.keras.optimizers import Adam

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

    def create_actor_model(self, state_input, root_net, action_dim, lr=0.001):
        """
        net = Dense(64, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01), activation="relu")(root_net)
        """
        net = Dense(64, activation="relu")(root_net)
        if self.use_batch_norm:
            net = BatchNormalization()(net)
        """
        net = Dense(64, input_dim=64,
                    kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01), activation="relu")(net)
        """
        net = Dense(128, activation="relu")(net)
        if self.use_batch_norm:
            net = BatchNormalization()(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        actor_out = Dense(action_dim,
                          #kernel_initializer="random_uniform",
                          activation='softmax',
                          #activation="relu",
                          #activity_regularizer=regularizers.l1(0.01),
                          name="actor_out")(net)

        #actor_out = Lambda(lambda x: x / tf.norm(x, ord=1))(actor_out)

        actor_model = Model(inputs=state_input, outputs=actor_out)
        adam = Adam(lr=lr, clipnorm=1., clipvalue=0.5)
        actor_model.compile(loss="mse", optimizer=adam)
        if DEBUG:
            print("ACTOR MODEL :", actor_model.summary())

        return actor_model

    def references(self):
        return self.state_input, self.net
