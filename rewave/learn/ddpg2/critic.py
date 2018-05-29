from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers import Input, Dense, BatchNormalization, Activation, Add, Concatenate

from tensorflow.python.keras.optimizers import Adam, RMSprop

from tensorflow.python.keras import metrics

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


    def create_critic_model(self, state_input, root_net, action_dim, lr=0.001):
        state_features = Dense(64, activation='relu')(root_net)
        state_features = Dense(32, activation='relu')(state_features)

        action_input = Input(shape=(action_dim,), name="action_input")
        action_features = Dense(32, activation='relu')(action_input)

        #merged = Add()([state_features, action_features])
        merged = Concatenate()([state_features, action_features])
        merged_h1 = Dense(48, activation='relu')(merged)

        critic_out = Dense(1, activation='linear', name="critic_output")(merged_h1)
        critic_model = Model(inputs=[state_input, action_input], outputs=critic_out)
        if DEBUG:
            print("CRITIC MODEL :", critic_model.summary())
        adam = Adam(lr=lr, clipnorm=1., clipvalue=0.5)
        critic_model.compile(loss="mean_squared_error", optimizer=adam, metrics={'critic_output': [metrics.mean_absolute_error, metrics.categorical_accuracy]})
        return action_input, critic_model

    def references(self):
        return self.state_input, self.action_input, self.net