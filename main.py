from __future__ import absolute_import
from datetime import date

import numpy as np

from rewave.environment.long_portfolio import PortfolioEnv

from rewave.learn.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from rewave.learn.ddpg_trainer import StockActor, StockCritic, DDPG, get_variable_scope, get_model_path, get_result_path, model_predictor, obs_normalizer

from rewave.learn.ddpg2.ddpg import DDPG as DDPG2
import rewave.marketdata.globaldatamatrix as gdm

from rewave.tools.data import normalize, prepare_dataframe

import tensorflow as tf

from tensorflow.python.keras.layers import Input
from tensorflow.python.keras import backend as K

def main():
    start_date = date(2008, 1, 1)
    end_date = date(2018, 1, 1)
    features_list = ['open', 'high', 'low', 'close']
    tickers_list = ['AAPL','ATVI','CMCSA','COST','CSX','DISH','EA','EBAY','FB','GOOGL','HAS','ILMN','INTC','MAR','REGN','SBUX']
    tickers_list = ['AAPL', 'ATVI', 'CMCSA', 'FB', 'HAS', 'ILMN', 'INTC',
                    'MAR', 'REGN', 'SBUX']

    historyManager = gdm.HistoryManager(tickers=tickers_list, online=True)
    df = historyManager.historical_data(start=start_date, end=end_date, tickers=tickers_list,
                                        features=features_list, adjusted=True)

    history = prepare_dataframe(df)
    history = history[:, :, :4]

    num_training_time = 1095

    # get target history
    target_history = np.empty(shape=(len(tickers_list), num_training_time, history.shape[2]))
    for i, stock in enumerate(tickers_list):
        target_history[i] = history[tickers_list.index(stock), :num_training_time, :]


    predictor_list = ['cnn', 'lstm']
    batch_norm_list = [False, True]
    window_length_list = [5, 7, 10, 15, 20]

    for predictor_type in predictor_list:
        for use_batch_norm in batch_norm_list:
            for window_length in window_length_list:

                # setup environment
                env = PortfolioEnv(start_date, end_date,
                                   window_length,
                                   tickers_list, features_list, batch_size=num_training_time)

                nb_classes = len(tickers_list) + 1
                nb_features = len(features_list)
                action_dim = [nb_classes]
                state_dim = [nb_classes, window_length]
                action_bound = 1.
                tau = 1e-3

                actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
                model_save_path = get_model_path(window_length, predictor_type, use_batch_norm)
                summary_path = get_result_path(window_length, predictor_type, use_batch_norm)

                variable_scope = get_variable_scope(window_length, predictor_type, use_batch_norm)

                tf.reset_default_graph()
                sess = tf.Session()

                with sess.as_default():

                    with tf.variable_scope(variable_scope):
                        root_net = None
                        state_inputs = Input(shape=([nb_classes, window_length, nb_features]))
                        root_net = model_predictor(state_inputs, predictor_type, use_batch_norm)

                        actor = StockActor(sess=sess, root_net=root_net, inputs=state_inputs, state_dim=state_dim,
                                           action_dim=action_dim, action_bound=action_bound,
                                           learning_rate=1e-4, tau=tau, batch_size=window_length,
                                           predictor_type=predictor_type, use_batch_norm=use_batch_norm)
                        critic = StockCritic(sess=sess, root_net=root_net, inputs=state_inputs, state_dim=state_dim,
                                             action_dim=action_dim, tau=1e-3,
                                             learning_rate=1e-3,
                                             predictor_type=predictor_type, use_batch_norm=use_batch_norm)
                        ddpg_model = DDPG(env=env, sess=sess, actor=actor, critic=critic, actor_noise=actor_noise,
                                          obs_normalizer="history",
                                          model_save_path=model_save_path,
                                          summary_path=summary_path)
                        ddpg_model.initialize(load_weights=True)
                        ddpg_model.train(debug=True)
                sess.close()

def main2():
    print("STRATING")
    start_date = date(2008, 1, 1)
    end_date = date(2017, 1, 1)
    features_list = ['open', 'high', 'low', 'close']
    tickers_list = ['AAPL', 'ATVI', 'CMCSA', 'COST', 'CSX', 'DISH', 'EA', 'EBAY', 'FB', 'GOOGL', 'HAS', 'ILMN', 'INTC',
                    'MAR', 'REGN', 'SBUX']

    predictor_list = ['lstm', 'cnn', 'dense']
    batch_norm_list = [False, True]
    window_length_list = [7, 10, 15, 20]

    for predictor_type in predictor_list:
        for use_batch_norm in batch_norm_list:
            for window_length in window_length_list:

                # setup environment
                num_training_time = 2000
                env = PortfolioEnv(start_date, end_date,
                                   window_length,
                                   tickers_list, features_list, batch_size=num_training_time)

                # variable_scope ="ddpg"
                action_dim = env.action_space.shape[0]
                actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

                model_save_path = get_model_path(window_length, predictor_type, use_batch_norm)
                summary_path = get_result_path(window_length, predictor_type, use_batch_norm)

                tf.reset_default_graph()
                sess = tf.Session()

                with sess.as_default():
                    # Missing this was the source of one of the most challenging an insidious bugs that I've ever encountered.
                    # it was impossible to save the model.
                    K.set_session(sess)

                    ddpg_model = DDPG2(env, sess, actor_noise, action_dim=action_dim, obs_normalizer="history",
                                       predictor_type=predictor_type, use_batch_norm=use_batch_norm,
                                       model_save_path= model_save_path, summary_path=summary_path)
                    ddpg_model.initialize(load_weights=False)
                    ddpg_model.train()

if __name__ == "__main__":
    main2()
    