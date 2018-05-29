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
from tensorflow.python.keras.layers import Add, Multiply, Lambda
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


def squeeze_middle2axes_operator( x4d ) :
    shape = x4d.shape # get dynamic tensor shape
    x3d = tf.reshape( x4d, [-1, shape[1].value * shape[2].value, shape[3].value ] )
    return x3d

def squeeze_middle2axes_shape( x4d_shape ) :
    in_batch, in_rows, in_cols, in_filters = x4d_shape
    if ( None in [ in_rows, in_cols] ) :
        output_shape = ( in_batch, None, in_filters )
    else :
        output_shape = ( in_batch, in_rows * in_cols, in_filters )
    return output_shape

def squeeze_first2axes_operator( x4d ) :
    shape = x4d.shape # get dynamic tensor shape
    x3d = tf.reshape( x4d, [shape[1].value * shape[1].value, shape[2].value, shape[3].value ] )
    return x3d

def squeeze_first2axes_shape( x4d_shape ) :
    in_batch, in_rows, in_cols, in_filters = x4d_shape
    if ( in_batch == None ) :
        output_shape = ( None, in_cols, in_filters )
    else :
        output_shape = ( in_batch*in_rows, in_cols, in_filters )
    return output_shape


def model_predictor(inputs, predictor_type, use_batch_norm):
    window_length = inputs.get_shape()[2]
    assert predictor_type in ['cnn', 'lstm'], 'type must be either cnn or lstm'
    assert window_length >= 3, 'window length must be at least 3'
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
        window_length = inputs.get_shape()[2]
        features = inputs.get_shape()[3]

        hidden_dim = 32

        if DEBUG:
            print('Shape input:', inputs.shape)
        #net = Lambda(squeeze_middle2axes_operator, output_shape=squeeze_middle2axes_shape)(inputs)
        #net = Lambda(squeeze_first2axes_operator, output_shape=squeeze_first2axes_shape)(inputs)

        net = Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))(inputs)
        if DEBUG:
            print('Shape input after transpose:', net.shape)

        #net = Reshape((window_length, features))(inputs)
        net = Reshape((window_length, num_stocks*features))(net)

        #net = tf.squeeze(inputs, axis=0)
        # reorder
        #net = tf.transpose(net, [1, 0, 2])
        if DEBUG:
            print('Reshaped input:', net.shape)
        net = LSTM(hidden_dim)(net)
        if DEBUG:
            print('After LSTM:', net.shape)
        #net = Reshape((num_stocks, hidden_dim))(inputs)
        if DEBUG:
            print('Output:', net.shape)
    else:
        raise NotImplementedError

    return net

class StockActor(ActorNetwork):
    def __init__(self, sess, root_net, inputs, state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
                 predictor_type, use_batch_norm):
        self.root_net = root_net
        self.inputs = inputs
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        ActorNetwork.__init__(self, sess, root_net, inputs, state_dim, action_dim, action_bound, learning_rate, tau, batch_size)

    def create_actor_network(self, root_net=None):
        """
        self.s_dim: a list specifies shape
        """

        nb_classes, window_length = self.s_dim
        assert nb_classes == self.a_dim[0]
        assert window_length > 2, 'This architecture only support window length larger than 2.'
        #inputs = Input(shape=(self.s_dim + [1]), name="input")

        net = root_net

        if root_net == None:
            net = model_predictor(self.inputs, self.predictor_type, self.use_batch_norm)

        net = Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(net)
        if self.use_batch_norm:
            net = BatchNormalization()(net)
        net = Activation("relu")(net)
        net = Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(net)
        if self.use_batch_norm:
            net = BatchNormalization()(net)
        net = Activation("relu")(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        out = Dense(self.a_dim[0], kernel_initializer="random_uniform", activation='softmax', name="actor_out")(net)

        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return out, scaled_out

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
    def __init__(self, sess, root_net, inputs, state_dim, action_dim, learning_rate, tau,
                 predictor_type, use_batch_norm):
        self.root_net = root_net
        self.inputs = inputs
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        CriticNetwork.__init__(self, sess, root_net, inputs, state_dim, action_dim, learning_rate, tau)

    def create_critic_network(self, root_net=None):
        #inputs = Input(shape=(self.s_dim + [1]))
        action = Input(self.a_dim)

        net = root_net

        if root_net == None:
            net = model_predictor(self.inputs, self.predictor_type, self.use_batch_norm)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = Dense(64, activation="relu")(net)
        t2 = Dense(64, activation="relu")(action)

        """
        t1 = Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(net)
        if self.use_batch_norm:
            t1 = BatchNormalization()(t1)
        t1 = Activation("relu")(t1)

        t2 = Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(action)
        if self.use_batch_norm:
            t2 = BatchNormalization()(t2)
        t2 = Activation("relu")(t2)
        """


        net = Add()([t1, t2])
        if self.use_batch_norm:
            net = BatchNormalization()(net)
        net = Activation("relu")(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        out = Dense(1, kernel_initializer="random_uniform",
                    activation='softmax', name="critic_out")(net)
        return action, out

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


def obs_normalizer_full(observation):
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
    observation = observation[:, :, :] / observation[:, :, 0]
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
    tickers_list = ['AAPL','ATVI','CMCSA','COST','CSX','DISH','EA','EBAY','FB','GOOGL','HAS','ILMN','INTC','MAR','REGN','SBUX']

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
    nb_features = len(features_list)

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

        root_net = None
        state_inputs = Input(shape=([nb_classes, window_length, nb_features]))
        root_net = model_predictor(state_inputs, predictor_type, use_batch_norm)

        root_model = Model(state_inputs, root_net)
        tf.summary.histogram("root_net", root_net)

        actor = StockActor(sess=sess, root_net=root_net, inputs=state_inputs, state_dim=state_dim, action_dim=action_dim, action_bound=action_bound,
                            learning_rate=1e-4, tau=tau, batch_size=batch_size,
                            predictor_type=predictor_type, use_batch_norm=use_batch_norm)
        critic = StockCritic(sess=sess, root_net=root_net, inputs=state_inputs, state_dim=state_dim, action_dim=action_dim, tau=1e-3,
                             learning_rate=1e-3,
                             predictor_type=predictor_type, use_batch_norm=use_batch_norm)
        ddpg_model = DDPG(env=env, sess=sess, actor=actor, critic=critic, actor_noise=actor_noise,
                          obs_normalizer="history",
                          model_save_path=model_save_path,
                          summary_path=summary_path)
        ddpg_model.initialize(load_weights=False)
        ddpg_model.train(debug=True)
