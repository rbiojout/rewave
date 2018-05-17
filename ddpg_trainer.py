"""
Use DDPG to train a stock trader based on a window of history price
"""

from __future__ import absolute_import, print_function, division

from rewave.learn.ddpg.actor import ActorNetwork
from rewave.learn.ddpg.critic import CriticNetwork
from rewave.learn.ddpg.ddpg import DDPG
from rewave.learn.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

from rewave.environment.long_portfolio import PortfolioEnv
from rewave.tools.data import normalize, prepare_dataframe


from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Reshape, LSTM
from tensorflow.python.keras.layers import Add, Multiply
from tensorflow.python.keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras import regularizers
from tensorflow.python.keras import backend as K

import rewave.marketdata.globaldatamatrix as gdm

from datetime import date

import numpy as np
#import tflearn
import tensorflow as tf
import argparse
import pprint

import random
from collections import deque

DEBUG = True


class ActorCritic:
    def __init__(self, env, sess, learning_rate = 0.001, epsilon = 1.0, epsilon_decay = .995, gamma = .05, tau = .125):
        self.env = env
        self.sess = sess

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.memory = deque(maxlen=2000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, self.env.action_space.shape[
                                                    0]])  # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, \
        self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)  # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        """
        state_input = Input(shape=self.env.observation_space.spaces['history'].shape)
        flatten = Flatten()(state_input)
        h1 = Dense(24, activation='relu')(flatten)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.env.action_space.shape[0], activation='relu')(h3)

        model = Model(inputs=state_input, outputs=output)
        """

        state_input = Input(shape=self.env.observation_space.spaces['history'].shape)
        conv2d_1 = Conv2D(filters=32, kernel_size=(1, 3), input_shape=env.observation_space.spaces['history'].shape,
                          padding='same',
                          data_format='channels_last',
                          activation='tanh')(state_input)
        conv2D_2 = Conv2D(64, (1, 3), padding='same', activation='tanh')(conv2d_1)
        maxpool_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2D_2)
        dropout_1 = Dropout(0.25)(maxpool_1)
        flatten = Flatten()(dropout_1)
        dense1 = Dense(128, activation='relu')(flatten)
        dropout_2 = Dropout(0.25)(dense1)
        output = Dense(self.env.action_space.shape[0], activation='relu')(dropout_2)
        model = Model(inputs=state_input, outputs=output)

        if DEBUG:
            print("ACTOR MODEL :", model.summary())
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.spaces['history'].shape)
        flatten = Flatten()(state_input)
        state_h1 = Dense(24, activation='relu')(flatten)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.env.action_space.shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(inputs=[state_input, action_input], outputs=output)
        print("CRITIC MODEL :", model.summary())
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], reward, verbose=0, batch_size=1)

    def train(self):
        batch_size = 64
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        weights = self.actor_model.get_weights()
        target_weights = self.target_actor_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_actor_model.set_weights(target_weights)


    def _update_critic_target(self):
        weights = self.critic_model.get_weights()
        target_weights = self.critic_target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.critic_target_model.set_weights(target_weights)


    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor_model.predict(cur_state).reshape((env.action_space.shape[0],))

def obs_normalizer(observation):
    """ Preprocess observation obtained by environment

    Args:
        observation: (nb_classes, window_length, num_features) or with info

    Returns: normalized

    """
    if isinstance(observation, tuple):
        observation = observation[0]
    if isinstance(observation, dict):
        observation = observation['history']
    # directly use close/open ratio as feature
    observation = observation[:, :, 3:4] / observation[:, :, 0:1]
    observation = normalize(observation)
    return observation


def obs_reshaper(observation):
    if isinstance(observation, tuple):
        observation = observation[0]
    if isinstance(observation, dict):
        observation = observation['history']
    return observation.reshape((1,)+observation.shape)

def mainold():
    sess = tf.Session()
    K.set_session(sess)

    start_date = date(2008, 1, 1)
    end_date = date(2018, 1, 1)
    features_list = ['open', 'high', 'low', 'close']
    tickers_list = ['AAPL', 'A']

    batch_size = 200

    env = PortfolioEnv(
        start_date=start_date,
        end_date=end_date,
        features_list=features_list,
        tickers_list=tickers_list,
        batch_size=batch_size,
        scale=True,
        buffer_bias_ratio=1e-6,
        trading_cost=0.00,
        window_length=window_length,
        output_mode='EIIE',
    )
    actor_critic = ActorCritic(env, sess)

    num_trials = 200
    trial_len = batch_size

    for trial in range(num_trials):
        print("----------- TRIAL ------- ",trial)
        cur_state = env.reset()
        action = env.action_space.sample()
        for step in range(trial_len):
            env.render()
            cur_state = obs_reshaper(cur_state)
            action = actor_critic.act(cur_state)

            new_state, reward, done, infos = env.step(action)
            action = action.reshape((1, env.action_space.shape[0]))
            #print("action shape:", action.shape)

            new_history = obs_reshaper(new_state)

            reward = reward.reshape((1,))
            actor_critic.remember(cur_state, action, reward, new_history, done)
            actor_critic.train()

            cur_state = new_state
            #print("===========================================")
            if done:
                print("done with step ", step, " action :", action, " infos date ", infos['date'])
                cur_state = env.reset()
                action = env.action_space.sample()
                break



def get_model_path(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return 'weights/stock/{}/window_{}/{}/checkpoint.ckpt'.format(predictor_type, window_length, batch_norm_str)


def get_result_path(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return 'results/stock/{}/window_{}/{}/'.format(predictor_type, window_length, batch_norm_str)


def get_variable_scope(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return '{}_window_{}_{}'.format(predictor_type, window_length, batch_norm_str)



def model_predictor(inputs, predictor_type, use_batch_norm):
    window_length = inputs.get_shape()[1]
    assert predictor_type in ['cnn', 'lstm'], 'type must be either cnn or lstm'
    if predictor_type == 'cnn':
        net = Conv2D(filters=32, kernel_size=(1, 3),
                          padding='same',
                          data_format='channels_last')(inputs)
        if use_batch_norm:
            net = BatchNormalization()(net)
        net = Activation("relu")(net)
        net = Conv2D(filters=32, kernel_size=(1, window_length - 2),
                     padding='valid',
                     data_format='channels_last')(net)
        if use_batch_norm:
            net = BatchNormalization()(net)
        net = Activation("relu")(net)
        if DEBUG:
            print('After conv2d:', net.shape)
        net = Flatten()(net)
        if DEBUG:
            print('Output:', net.shape)
    elif predictor_type == 'lstm':
        num_stocks = inputs.get_shape()[1]
        hidden_dim = 32
        net = Reshape((-1, window_length, 1))(inputs)
        if DEBUG:
            print('Reshaped input:', net.shape)
        net = LSTM(hidden_dim)(net)
        if DEBUG:
            print('After LSTM:', net.shape)
        net = Reshape((-1, num_stocks, hidden_dim))(inputs)
        if DEBUG:
            print('After reshape:', net.shape)
        net = Flatten()(net)
        if DEBUG:
            print('Output:', net.shape)
    else:
        raise NotImplementedError

    return net

class StockActor(ActorNetwork):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
                 predictor_type, use_batch_norm):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        ActorNetwork.__init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size)

    def create_actor_network(self):
        """
        self.s_dim: a list specifies shape
        """
        nb_classes, window_length = self.s_dim
        assert nb_classes == self.a_dim[0]
        assert window_length > 2, 'This architecture only support window length larger than 2.'
        inputs = Input(shape=(self.s_dim + [1]), name="input")

        net = model_predictor(inputs, self.predictor_type, self.use_batch_norm)
        print("NET ACTOR 1 ", net)

        net = Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(net)
        if self.use_batch_norm:
            net = BatchNormalization()(net)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = Activation("relu")(net)
        net = Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(net)
        if self.use_batch_norm:
            net = BatchNormalization()(net)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = Activation("relu")(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        out = Dense(self.a_dim[0], kernel_initializer="random_uniform", activation='softmax', name="actor_out")(net)

        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        print("NET ACTOR ",net)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })


class StockCritic(CriticNetwork):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars,
                 predictor_type, use_batch_norm):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        CriticNetwork.__init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars)

    def create_critic_network(self):
        inputs = Input(shape=(self.s_dim + [1]))
        action = Input(self.a_dim)

        net = model_predictor(inputs, self.predictor_type, self.use_batch_norm)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = Dense(64)(net)
        t2 = Dense(64)(action)

        net = Add()([t1, t2])
        if self.use_batch_norm:
            net = BatchNormalization()(net)
        net = Activation("relu")(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        out = Dense(1, kernel_initializer="random_uniform",
                    activation='softmax', name="critic_out")(net)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })


def obs_normalizer(observation):
    """ Preprocess observation obtained by environment

    Args:
        observation: (nb_classes, window_length, num_features) or with info

    Returns: normalized

    """
    if isinstance(observation, tuple):
        observation = observation[0]
    if isinstance(observation, dict):
        observation = observation['history']
    # directly use close/open ratio as feature
    observation = observation[:, :, 3:4] / observation[:, :, 0:1]
    observation = normalize(observation)
    return observation


def test_model(env, model):
    observation, info = env.reset()
    done = False
    while not done:
        action = model.predict_single(observation)
        observation, _, done, _ = env.step(action)
    env.render()


def test_model_multiple(env, models):
    observation, info = env.reset()
    done = False
    while not done:
        actions = []
        for model in models:
            actions.append(model.predict_single(observation))
        actions = np.array(actions)
        observation, _, done, info = env.step(actions)
    env.render()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Provide arguments for training different DDPG models')

    parser.add_argument('--debug', '-d', help='print debug statement', default=False)
    parser.add_argument('--predictor_type', '-p', help='cnn or lstm predictor', required=True)
    parser.add_argument('--window_length', '-w', help='observation window length', required=True)
    parser.add_argument('--batch_norm', '-b', help='whether to use batch normalization', required=True)

    args = vars(parser.parse_args())

    pprint.pprint(args)

    if args['debug'] == 'True':
        DEBUG = True
    else:
        DEBUG = False

    start_date = date(2008, 1, 1)
    end_date = date(2018, 1, 1)
    features_list = ['open', 'high', 'low', 'close']
    tickers_list = ['AAPL', 'A']

    historyManager = gdm.HistoryManager(tickers=tickers_list, online=True)
    df = historyManager.historical_data(start=start_date, end=end_date, tickers=tickers_list,
                                        features=features_list, adjusted=True)

    history = prepare_dataframe(df)
    abbreviation = tickers_list
    history = history[:, :, :4]
    target_stocks = abbreviation
    num_training_time = 1095
    window_length = int(args['window_length'])
    nb_classes = len(target_stocks) + 1

    # get target history
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[abbreviation.index(stock), :num_training_time, :]

    # setup environment

    env = PortfolioEnv(start_date, end_date,
                       window_length,
                       tickers_list, features_list, batch_size=num_training_time)

    action_dim = [nb_classes]
    state_dim = [nb_classes, window_length]
    batch_size = 64
    action_bound = 1.
    tau = 1e-3
    assert args['predictor_type'] in ['cnn', 'lstm'], 'Predictor must be either cnn or lstm'
    predictor_type = args['predictor_type']
    if args['batch_norm'] == 'True':
        use_batch_norm = True
    elif args['batch_norm'] == 'False':
        use_batch_norm = False
    else:
        raise ValueError('Unknown batch norm argument')
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    model_save_path = get_model_path(window_length, predictor_type, use_batch_norm)
    summary_path = get_result_path(window_length, predictor_type, use_batch_norm)

    variable_scope = get_variable_scope(window_length, predictor_type, use_batch_norm)

    with tf.variable_scope(variable_scope):
        sess = tf.Session()

        actor = StockActor(sess=sess, state_dim=state_dim, action_dim=action_dim, action_bound=action_bound,
                            learning_rate=1e-4, tau=tau, batch_size=batch_size,
                            predictor_type=predictor_type, use_batch_norm=use_batch_norm)
        critic = StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=1e-3,
                             learning_rate=1e-3, num_actor_vars=actor.get_num_trainable_vars(),
                             predictor_type=predictor_type, use_batch_norm=use_batch_norm)
        ddpg_model = DDPG(env=env, sess=sess, actor=actor, critic=critic, actor_noise=actor_noise,
                          obs_normalizer=obs_normalizer,
                          model_save_path=model_save_path,
                          summary_path=summary_path)
        ddpg_model.initialize(load_weights=False)
        ddpg_model.train(debug=True)
