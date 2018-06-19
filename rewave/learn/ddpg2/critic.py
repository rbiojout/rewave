from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers import Input, Dense, BatchNormalization, Activation, Add, Concatenate, Flatten, Dropout, Multiply, AveragePooling2D, Lambda

from tensorflow.python.keras.optimizers import Adam, RMSprop

from tensorflow.python.keras import regularizers

from tensorflow.python.keras import metrics

from tensorflow.python.keras import backend as K


DEBUG = True

class Critic(object):
    def __init__(self, state_input, root_net, action_dim, lr=0.001):
        """

        :param input_state: the input state layer for the actor
        :param root_net: the network shared between actor and critic
        """
        self.state_input = state_input
        self.root_net = root_net
        self.action_dim = action_dim
        self.lr = lr

        self.action_input, self.net = self.create_critic_model(self.state_input, self.root_net, self.action_dim, self.lr)

    def create_critic_model(self, state_input, root_net, action_dim, lr=0.001, dropout=0.3):
        """
        state_input_permuted = Lambda(lambda x: K.permute_dimensions(x, (0, 3, 1, 2)))(state_input)
        nb_features = state_input.shape[3]
        root_net = AveragePooling2D(pool_size=(nb_features,1),strides=None, padding='same')(state_input_permuted)
        """

        root_net = Flatten()(root_net)
        state_features = Dense(400, activation='selu')(root_net)
        #state_features = Dropout(dropout)(state_features)

        action_input = Input(shape=(action_dim,), name="action_input")

        #merged = Add()([state_features, action_features])
        merged = Concatenate()([state_features, action_input])

        merged = Dense(300, activation='selu')(merged)
        #merged = Dropout(dropout)(merged)

        critic_out = Dense(1, name="critic_output")(merged)
        critic_model = Model(inputs=[state_input, action_input], outputs=critic_out)
        if DEBUG:
            print("CRITIC MODEL :", critic_model.summary())
        adam = Adam(lr=lr)
        critic_model.compile(loss="mean_squared_error",
                             optimizer=adam,
                             metrics={'critic_output': [metrics.mean_absolute_error, metrics.categorical_accuracy]}
                            )

        return action_input, critic_model

    def references(self):
        return self.state_input, self.action_input, self.net