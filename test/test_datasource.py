from __future__ import absolute_import
from unittest import TestCase
import unittest
from rewave.marketdata.replaybuffer import ReplayBuffer, Experience
from datetime import date
import logging

import numpy as np

from rewave.environment.long_portfolio import DataSrc

class TestDataSrc(unittest.TestCase):
    def setUp(self):
        self.start_date = date(2017, 1, 1)
        self.end_date = date(2018, 1, 1)
        self.features_list = ['close', 'high', 'low']
        self.tickers_list = ['AAPL', 'MSFT', 'A']
        self.batch_size = 30
        self.scale = True
        self.scale_extra_cols = True
        self.buffer_bias_ratio = 1e-5
        self.window_length = 50
        self.is_permed = False
        self.src = DataSrc(start_date=self.start_date, end_date=self.end_date,
                           tickers_list= self.tickers_list,features_list=self.features_list,
                           batch_size=self.batch_size, scale=self.scale,
                           scale_extra_cols=self.scale_extra_cols,
                           buffer_bias_ratio=self.buffer_bias_ratio,
                           window_length=self.window_length,is_permed=self.is_permed)

    def test_params(self):
        src = self.src

        self.assertEquals(self.scale, src.scale, "scale is not set correctly")
        self.assertEquals(self.scale_extra_cols, src.scale_extra_cols, "scale_extra_cols is not set correctly")
        self.assertEquals(self.window_length, src.window_length, "window length is not set correctly")
        self.assertEquals(self.batch_size, src.batch_size, "batch size is not set correctly")
        self.assertListEqual(self.features_list, src.features, "asset names not set correctly")
        self.assertListEqual(['cash']+ self.tickers_list, src.asset_names, "asset names not set correctly")

    def test_data(self):
        data = self.src._data

        self.assertEqual(len(self.tickers_list)+1, data.shape[0], "dimension 0 must be the assets")
        self.assertEqual(len(self.features_list), data.shape[2], "dimension 2 must be the features")
        self.assertEqual(0, np.count_nonzero(np.isnan(self.src._data)), "n/a values are not possible")
        self.assertEqual(1.0, data[0,0,0], "cash must be at the begining")
        self.assertEqual(1.0, data[0, :, 0].sum()/data.shape[1], "cash must be 1.0 for all the periods")
        self.assertEqual(1.0, data[0, :, :].sum() / data.shape[1] / data.shape[2], "cash must be 1.0 for all the periods and all features")

    def test_reset(self):
        src = self.src

        indices = src.indices
        window_length = src.window_length
        asset_names = src.asset_names
        # PVM back to initialized state
        src.reset()
        PVM = src.PVM
        self.assertEqual(len(indices),PVM.shape[0], "number of times should be aligned")
        self.assertEqual(len(asset_names), PVM.shape[1], "number of assets should be aligned")
        self.assertEqual(1.0 / len(asset_names), PVM.iloc[0, 0], "initialization should be uniform")
        self.assertEqual(1.0, PVM.iloc[0,:].sum(), "weights should sum to 1")
        self.assertEqual(1.0, PVM.iloc[:, :].sum().sum()/PVM.shape[0], "weights should sum to 1")
        self.assertEqual(src.step, src.batch[0], "first index is begining of batch")
        self.assertGreaterEqual(src.step, 0, "batch must start with step greater or equal to 0")
        self.assertEqual(src.batch_size, src.batch[-1] - src.batch[0] +1 , "batch size is not correct")


    def test_step(self):
        initial_step= self.src.step
        X, y, last_w, setw, open_history, done = self.src._step()
        self.assertEqual(False, done, "batch should continue")
        next_step = self.src.step
        self.assertEqual(1, next_step - initial_step, "step must grow by 1")
        self.assertEqual(len(self.src.asset_names), X.shape[0], "X shape 0: dimension must be assets length")
        self.assertEqual(self.src.window_length, X.shape[1], "X shape 1: time sequence length must be window_length")
        self.assertEqual(len(self.src.features), X.shape[2], "X shape 2: dimension must be features length")
        self.assertEqual((len(self.src.asset_names),), y.shape, "y shape: dimension must be assets length")
        self.src.reset()
        initial_step = self.src.step
        for i in range(0, self.src.batch_size-1):
            X, y, last_w, setw, open_history, done = self.src._step()
        self.assertEqual(initial_step + self.src.batch_size - 1, self.src.step, "step must end at batch_size -1")
        self.assertEqual(True, done, "batch should finish")

    def test_keep_last(self):
        """
        because we take the previous data, we can only start at one
        :return:
        """
        src = DataSrc(start_date=self.start_date, end_date=self.end_date,
                           tickers_list=self.tickers_list, features_list=self.features_list,
                           batch_size=0, scale=self.scale,
                           scale_extra_cols=self.scale_extra_cols,
                           buffer_bias_ratio=self.buffer_bias_ratio,
                           window_length=self.window_length, is_permed=self.is_permed)
        src.reset()
        self.assertEqual(0, src.step, "step equal zero in reset")
        src._step()
        self.assertEqual(1, src.step, "step start at one")