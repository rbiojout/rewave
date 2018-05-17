################################################################################
# Description: Financial market environment for the automatic asset allocation
#       can not be short
#              task. It is based on the PyBrain architecture.
# Author:      Raphael Biojout
# Email:       rbiojout@gmail.com
# Date:        02/05/2018
################################################################################

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pprint import pprint

import os
import time

import logging
import sqlite3
from datetime import date, datetime

#from typing import Any, Tuple, Dict

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import rewave.marketdata.globaldatamatrix as gdm
import rewave.marketdata.replaybuffer as rb

from rewave.tools.configprocess import parse_time, parse_list
from rewave.tools.data import random_shift
from rewave.tools.data import dataframe_fillna, prepare_dataframe

from rewave.constants import eps

from rewave.callbacks.notebook_plot import LivePlotNotebook

from rewave.environment.normalized_box import NormalizedBox


BATCH_SIZE = 30
WINDOW_LENGTH = 20
BUFFER_BIAS_RATIO = 1e-5
DELTA_MU = 1e-10
NB_ACTIONS = 10

logger = logging.getLogger(__name__)

class DataSrc(object):
    """Acts as data provider for each new episode."""

    def __init__(self, start_date, end_date,
                 tickers_list,
                 features_list,
                 batch_size=BATCH_SIZE,
                 scale=True, scale_extra_cols=True,
                 buffer_bias_ratio=BUFFER_BIAS_RATIO, window_length=WINDOW_LENGTH, is_permed=False):
        """
        DataSrc.
        construct a 3D array ticker x time x features
        from the data collected in the global period start / end
        start_date: first date for collecting data
        end_date: last date for collecting data
        tickers_list: array with the names of the tickers, cash is added in asset_names
        features_list: array with the features used
        batch_size: total steps in episode
        scale: boolean regarding scale the data for each episode
        buffer_bias: used for the random factor


        """

        self.scale = scale
        self.scale_extra_cols = scale_extra_cols
        self.window_length = window_length
        # calculate the total number of steps
        self.batch_size = batch_size

        # recover from DataBase
        historyManager = gdm.HistoryManager(tickers=tickers_list, online=True)
        df = historyManager.historical_data(start=start_date, end=end_date, tickers=tickers_list,
                                                   features=features_list, adjusted=True)

        # add the cash
        cash_header = pd.MultiIndex.from_product([['cash'], features_list],
                                                 names=['ticker', 'feature'])
        cash = pd.DataFrame(1, index=df.index, columns=cash_header)
        df = pd.concat([cash, df], axis=1)

        # order columns
        tickers = ['cash'] + tickers_list
        df = df.reindex(columns=tickers, level=0)
        df = df.reindex(columns=features_list, level=1)

        # get the last close from the index before
        # when considering continuous trading the close price of the previous period is equal to the opening
        # get the open price considering that the portfolio allocation is appening at the begining of the period
        open_history = historyManager.historical_data(start=start_date, end=end_date, tickers=tickers_list,
                                                   features=['open'], adjusted=True)

        # add the cash
        cash_header = pd.MultiIndex.from_product([['cash'], ['open']],
                                                 names=['ticker', 'feature'])
        cash_open = pd.DataFrame(1, index=df.index, columns=cash_header)

        open_history = pd.concat([cash_open, open_history], axis=1)

        # don't consider the window_length
        open_history = open_history[window_length:]
        open_history = dataframe_fillna(open_history, "both")
        open_history = prepare_dataframe(open_history)
        self.open_history = open_history.reshape(open_history.shape[0], open_history.shape[1])

        # keep the history dataframe
        # not the adjusted price in this case and only the close
        self.history = historyManager.historical_data(start=start_date, end=end_date, tickers=tickers_list,
                                                   features=['open', 'high', 'low','close', 'volume'], adjusted=False)

        self.history = self.history[window_length:]

        # get rid of NaN's
        df = dataframe_fillna(df, "both")

        self.asset_names = df.columns.levels[0].tolist()
        self.features = df.columns.levels[1].tolist()
        self._times = df.index

        # dataframe to matrix with the right order in dimensions
        self._data = prepare_dataframe(df)

        # keep track of indices
        self.indices = np.arange(len(df)-window_length)

        # special case for batch_size=0 (all steps)
        # store the number of steps
        if batch_size == 0:
            self.batch_size = len(self.indices)

        # Portfolio Vector Memory
        # initialize with reset
        self.PVM = pd.DataFrame()

        self.price_columns = ['close', 'high', 'low', 'open']
        self.non_price_columns = set(
            df.columns.levels[1]) - set(self.price_columns)

        # Stats to let us normalize non price columns
        if scale_extra_cols:
            x = self._data.reshape((-1, len(self.features)))
            self.stats = dict(mean=x.mean(0), std=x.std(0))
            # for column in self._data.columns.levels[1].tolist():
            #     x = df.xs(key=column, axis=1, level='Price').as_matrix()[:, :]
            #     self.stats["mean"].append(x.mean())
            #      = dict(mean=x.mean(), std=x.std())

        self._replay_buffer = rb.ReplayBuffer(start_index=self.indices[0],
                                               end_index=self.indices[-1],
                                               sample_bias=buffer_bias_ratio,
                                               batch_size=self.batch_size,
                                               is_permed=is_permed)

        # batch of indexes
        # initialized with reset
        self.batch = []
        self.step = 0

        self.reset()


    def _step(self):
        """
        @:return: the next batch of training sample. The sample is a dictionary
        with key "X"(input data); "y"(future relative price); "last_w" a numpy array
        with shape [batch_size, assets]; "w" a list of numpy arrays list length is
        batch_size
        """

        last_w = self.PVM.values[self.step - 1, :]

        def setw(w):
            self.PVM.iloc[self.step, :] = w

        M = self.get_submatrix(self.step)
        M = np.array(M)
        X = M[ :, :-1, :]
        y = M[ :, -1, 0] / M[ 0, None, -2, 0]

        # @TODO put in return
        open_history =[]
        try:
            open_history = self.open_history[:, self.step:self.step +1].reshape((self.open_history.shape[0],))
        except ValueError:
            print("ValueError for step ", self.step)
        #history = self.history[:, self.step:self.step + self.window_length +1].copy()

        self.step += 1

        done = bool(self.step >= self.batch[-1] -1)

        return X, y, last_w, setw, open_history, done



    def _pack_samples(self, indexs):
        """
        NOT USED
        return the elements for the indexs
        :param indexs: indexs from which we extract
        :return: inputs, outputs, control functions
        """
        indexs = np.array(indexs)

        last_w = self.PVM.values[indexs-1, :]

        def setw(w):
            self.PVM.iloc[indexs, :] = w
        M = [self.get_submatrix(index) for index in indexs]
        M = np.array(M)
        X = M[:, :, :-1, :]
        y = M[:, :, -1, :] / M[:, 0, None, -2, :]

        return {"X": X, "y": y, "last_w": last_w, "setw": setw}


    def get_submatrix(self, ind):
        # get history matrix from dataframe
        # returns
        # - history array with shape : tichers, window, features
        # - y vector
        # - done boolean


        # we recover the sequence including the value after the squence window
        sequence = self._data[:, ind:ind + self.window_length +1].copy()

        # close is the first feature
        last_close_price = sequence[:, -2, 0]

        if self.scale:
            sequence /= last_close_price[:, np.newaxis, np.newaxis]

        nb_pc = len(self.price_columns)
        if self.scale_extra_cols:
            # normalize non price columns
            sequence[:, :, nb_pc:] -= self.stats["mean"][None, None, nb_pc:]
            sequence[:, :, nb_pc:] /= self.stats["std"][None, None, nb_pc:]
            sequence[:, :, nb_pc:] = np.clip(
                sequence[:, :, nb_pc:],
                self.stats["mean"][nb_pc:] - self.stats["std"][nb_pc:] * 10,
                self.stats["mean"][nb_pc:] + self.stats["std"][nb_pc:] * 10
            )

        sequence = np.around(sequence, 1)


        return sequence

        """
        # (eq.1) prices with cash at first position
        y1 = sequence[:,-1,0]

        self.step += 1
        history = sequence[:,:-1,:]
        done = bool(self.step >= self.batch_size)

        return history, y1, done
        """

    def reset(self):
        # reset the memory
        self.PVM = pd.DataFrame(index=self._times[self.window_length:], columns=self.asset_names)
        self.PVM = self.PVM.fillna(1.0 / len(self.asset_names))

        self.batch = [exp.state_index for exp in self._replay_buffer.next_experience_batch()]
        self.step = self.batch[0]

class PortfolioSim(object):
    """
    Portfolio management sim.

    Params:
    - cost e.g. 0.0025 is max in Poliniex

    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, tickers_list=[], batch_size=BATCH_SIZE, trading_cost=0.0025, time_cost=0.0):
        self.cost = trading_cost
        self.time_cost = time_cost
        self.batch_size = batch_size
        self.tickers_list = tickers_list
        self.infos = []
        self.w0 = np.array([1.0] + [0.0] * len(self.tickers_list))
        self.p0 = 1.0
        self.reset()
        logging.basicConfig(filename='environment.log', filemode='w', level=logging.DEBUG)

    def calculate_pv_after_commission(self, w1, w0, commission_rate):
        """
        @:param w1: target portfolio vector, first element is cash
        @:param w0: rebalanced last period portfolio vector, first element is cash
        @:param commission_rate: rate of commission fee, proportional to the transaction cost
        """
        if commission_rate != 0.0:
            print("COMMISSION RATE :", commission_rate)
        mu0 = 1
        mu1 = 1 - 2 * commission_rate + commission_rate ** 2
        while abs(mu1 - mu0) > DELTA_MU:
            mu0 = mu1
            mu1 = (1 - commission_rate * w0[0] -
                   (2 * commission_rate - commission_rate ** 2) *
                   np.sum(np.maximum(w0[1:] - mu1 * w1[1:], 0))) / \
                  (1 - commission_rate * w1[0])
        return mu1

    def _step(self, w1, y1):
        """
        Step.
        w0 - last action of portfolio weights
        p0 - last portfolio total value
        w1 - new action of portfolio weights - e.g. [0.1,0.9, 0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        w0 = self.w0
        p0 = self.p0

        # calculate the portfolio deformation (first element is cash)
        # impact of commission (mu)
        mu = self.calculate_pv_after_commission(w1, w0, self.cost)

        # portfolio value, total capital value
        p1 = mu * p0

        # evaluate the composition of the portfolio
        split_assets = np.multiply(p1, w1)
        # @TODO use the quote_price_previous = close_price/y1
        #shares = np.divide(split_assets, quote_price_previous)

        portfolio_change = mu * np.dot(w1, y1)

        """
        logging.info("the step is {} for {}".format(self._steps, self.time_index[self._steps]))
        logging.debug("the raw omega is {}".format(omega))
        future_price = np.concatenate((np.ones(1), self.__get_matrix_y()))
        # logging.debug("the future price vector is {}".format(future_price))
        quote_price_close = (self.__test_set["history_close"]).iloc[:, self._steps]
        quote_price_previous = np.concatenate( [np.ones(1),
                                np.divide(quote_price_close.values, self.__get_matrix_y()) ])
        # impact of commission (mu)
        pv_after_commission = calculate_pv_after_commission(omega, self._last_omega, self._commission_rate)
        self.__mu.append(pv_after_commission)
        
        # evaluate the shares of assets
        last_capital = self._total_capital*pv_after_commission
        split_assets = np.multiply(last_capital, omega)
        # logging.debug("the split of assets is {}".format(split_assets))
        shares = np.divide(split_assets, quote_price_previous)

        self.__shares_matrix[self._steps] = shares
        # logging.debug("the number of assets is {}".format(shares))

        portfolio_change = pv_after_commission * np.dot(omega, future_price)
        self._total_capital *= portfolio_change
        self._last_omega = pv_after_commission * omega * \
                           future_price /\
                           portfolio_change
        logging.debug("the portfolio change this period is : {}".format(portfolio_change))
        self.__test_pc_vector.append(portfolio_change)
        self.__test_omega.append(omega)
        """

        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # (eq7) weights evolve into

        # (eq16) cost to change portfolio
        # (excluding change in cash to avoid double counting for transaction cost)
        mu1 = self.cost * (
            np.abs(dw1[1:] - w1[1:])).sum()

        p1 = p0 * mu * np.dot(y1, w1)  # (eq11) final portfolio value

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        # can't have negative holdings in this model (no shorts)
        p1 = np.clip(p1, 0, np.inf)

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + eps) / (p0 + eps))  # (eq10) log rate of return
        # (eq22) immediate reward is log rate of return scaled by episode length
        # @TODO verify that we need to divide or not
        #reward = r1 / self.batch_size
        reward = r1 *1e4

        # remember for next step
        self.w0 = w1
        self.p0 = p1

        # if we run out of money, we're done
        done = bool(p1 == 0)

        # should only return single values, not list
        info = {
            "mu": mu,
            "reward": reward,
            "log_return": r1,
            "y_return":y1.mean(),
            "portfolio_value": p1,
            "portfolio_change": portfolio_change,
            "market_return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": mu1,
        }
        # record weights and prices
        for i, name in enumerate(['cash'] + self.tickers_list):
            info['weight_' + name] = w1[i]
            info['price_' + name] = y1[i]

        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.w0 = np.array([1.0] + [0.0] * len(self.tickers_list))
        self.p0 = 1.0

class DiscretePortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.

    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.

    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['notebook', 'ansi']}


    def __init__(self,
                 start_date, end_date,
                 window_length=WINDOW_LENGTH,
                 tickers_list=['AAPL'],
                 features_list=['close', 'open'],
                 # market,
                 batch_size=BATCH_SIZE,
                 buffer_bias_ratio = BUFFER_BIAS_RATIO,
                 online=True,
                 is_permed=False,
                 trading_cost=0.0025,
                 time_cost=0.00,
                 output_mode='EIIE',
                 log_dir=None,
                 scale=True,
                 scale_extra_cols=True,
                 ):
        """
        An environment for financial portfolio management.

        Params:
            df - csv for data frame index of timestamps
                 and multi-index columns levels=[['LTCBTC'],...],['open','low','high','close']]
            steps - steps in episode
            window_length - how many past observations["history"] to return
            batch_size for extracting the batches.
            buffer_bias_ratio random control for the batch. If defined to ZERO, start at the first index, if defined to ONE start at the end index
            trading_cost - cost of trade as a fraction,  e.g. 0.0025 corresponding to max rate of 0.25% at Poloniex (2017)
            time_cost - cost of holding as a fraction
            output_mode: decides observation["history"] shape
            - 'EIIE' for (assets, window, 3)
            - 'atari' for (window, window, 3) (assets is padded)
            - 'mlp' for (assets*window*3)
            log_dir: directory to save plots to
            scale - scales price data by last opening price on each episode (except return)
            scale_extra_cols - scales non price data using mean and std for whole dataset
        """

        self.start_date = start_date
        self.end_date = end_date
        self.tickers_list = tickers_list
        self.tickers_number = len(tickers_list)
        self.features_list = features_list
        self.features_number = len(features_list)
        self.window_length = window_length

        self.buffer_bias_ratio = buffer_bias_ratio
        if batch_size == 0:
            self.buffer_bias_ratio = 0
        self.scale_extra_cols = scale_extra_cols
        self.online = online
        self.is_permed = is_permed

        self.trading_cost = trading_cost
        self.time_cost = time_cost


        self.src = DataSrc(start_date=start_date, end_date=end_date,
                           tickers_list=tickers_list,
                           features_list=features_list,
                           scale=scale, window_length=window_length, scale_extra_cols=scale_extra_cols,
                           batch_size=batch_size,
                           buffer_bias_ratio=self.buffer_bias_ratio,
                           is_permed = is_permed)


        self.PVM = self.src.PVM

        self.batch_size = self.src.batch_size

        self._plot = self._plot2 = self._plot3 = None
        self.output_mode = output_mode
        self.sim = PortfolioSim(
            tickers_list=tickers_list,
            trading_cost=trading_cost,
            time_cost=time_cost,
            batch_size=batch_size)
        self.log_dir = log_dir

        # openai gym attributes
        # action will be the portfolio weights [cash_bias,w1,w2...] where wn are [0, 1] for each asset
        nb_assets = len(self.src.asset_names)
        self.action_space = gym.spaces.NormalizedBox(
            0, NB_ACTIONS, shape=(nb_assets,), dtype = np.float32 )

        # get the history space from the data min and max
        if output_mode == 'EIIE':
            obs_shape = (
                nb_assets,
                window_length,
                len(self.src.features)
            )
        elif output_mode == 'atari':
            obs_shape = (
                window_length,
                window_length,
                len(self.src.features)
            )
        elif output_mode == 'mlp':
            obs_shape = ((nb_assets) * window_length * \
                (len(self.src.features)),)
        else:
            raise Exception('Invalid value for output_mode: %s' %
                            self.output_mode)

        self.observation_space = gym.spaces.Dict({
            'history': gym.spaces.Box(
                -10,
                20 if scale else 1,  # if scale=True observed price changes return could be large fractions
                obs_shape,
                dtype=np.float32
            ),
            'weights': self.action_space
        })
        self._plot = self._plot2 = self._plot3 = None
        self.reset()

    def _on_market_data(self, end_date):
        self.end_date = end_date

        self.src = DataSrc(start_date=self.start_date, end_date=end_date,
                           tickers_list=self.tickers_list,
                           features_list=self.features_list,
                           scale=self.scale, window_length=self.window_length)

    def step(self, action):
        """
        Step the env.

        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight between 0 and 1. The first (w0) is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        logging.debug('action: %s', action)

        weights = np.clip(action/NB_ACTIONS, 0.0, 1.0)
        weights /= weights.sum() + eps

        # Sanity checks
        assert self.action_space.contains(
            action), 'action should be within %r but is %r' % (self.action_space, action)
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, decimal=3, err_msg='weights should sum to 1. action="%s"' % weights)

        # data gathered from the history
        history, y1, last_w, setw, open_history, done1 = self.src._step()

        # data gathered from the sim
        reward, info, done2 = self.sim._step(weights, y1)

        # add shares
        all_assets = ['cash'] + self.sim.tickers_list
        for index in range(0,len(all_assets)):
            name = all_assets[index]
            info['share_' + name] = info['portfolio_value'] * info['weight_' + name] / open_history[index]
            info['quote_' + name] = open_history[index]

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod(
            [inf["market_return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = self.src._times[self.src.step].timestamp()
        info['steps'] = self.src.step

        self.infos.append(info)

        # reshape history according to output mode
        if self.output_mode == 'EIIE':
            pass
        elif self.output_mode == 'atari':
            padding = history.shape[1] - history.shape[0]
            history = np.pad(history, [[0, padding], [
                0, 0], [0, 0]], mode='constant')
        elif self.output_mode == 'mlp':
            history = history.flatten()

        return dict(history= history, weights= weights), reward, done1 or done2, info

    def reset(self):
        self.src.reset()
        self.sim.reset()
        self.infos = []
        action = self.sim.w0
        self._plot = self._plot2 = self._plot3 = None
        observation, reward, done, info = self.step(action)
        return observation

    def seed(self, seed):
        np.random.seed(seed)
        return [seed]

    def df_info(self):
        df_info = pd.DataFrame(self.infos)
        df_info.index = pd.to_datetime(df_info["date"], unit='s')
        return df_info

    def render(self, mode='notebook', close=False):
        # if close:
            # return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'notebook':
            self.plot_notebook(close)

    def plot_notebook(self, close=False):
        """Live plot using the jupyter notebook rendering of matplotlib."""
        # @TODO fixme. For the moment, redraw from zero
        #self._plot = self._plot2 = self._plot3 = None

        if close:
            self._plot = self._plot2 = self._plot3 = None
            return

        df_info = pd.DataFrame(self.infos)
        #df_info.index = pd.to_datetime(df_info["date"], unit='s')
        x = df_info.index.to_pydatetime()

        # plot prices and performance
        _plot_dir = None
        all_assets = ['cash'] + self.sim.tickers_list
        if not self._plot:
            colors = [None] * len(all_assets) + ['black', 'grey']
            self._plot_dir = os.path.join(
                self.log_dir, 'notebook_plot_prices_' + str(time.time())) if self.log_dir else None
            self._plot = LivePlotNotebook(
                log_dir=self._plot_dir, title='prices & performance', labels=all_assets + ["Portfolio"], ylabel='value', colors=colors)
        y_portfolio = df_info["portfolio_value"]
        y_return = df_info["y_return"]
        y_assets = [df_info['price_' + name].cumprod()
                    for name in all_assets]
        self._plot.update(x, y_assets + [y_portfolio, y_return])


        # plot portfolio weights
        if not self._plot2:
            self._plot_dir2 = os.path.join(
                self.log_dir, 'notebook_plot_weights_' + str(time.time())) if self.log_dir else None
            self._plot2 = LivePlotNotebook(
                log_dir=self._plot_dir2, labels=all_assets, title='weights', ylabel='weight')
        ys = [df_info['weight_' + name] for name in all_assets]

        self._plot2.update(x, ys, max=100)


        # plot portfolio costs
        if not self._plot3:
            self._plot_dir3 = os.path.join(
                self.log_dir, 'notebook_plot_cost_' + str(time.time())) if self.log_dir else None
            self._plot3 = LivePlotNotebook(
                log_dir=self._plot_dir3, labels=['cost'], title='Commissions', ylabel='cost')
        ys = [df_info['cost'].cumsum()]
        ys = [(1-df_info['mu']).cumsum()]
        self._plot3.update(x, ys, max=100)


        if close:
            self._plot = self._plot2 = self._plot3 = None
