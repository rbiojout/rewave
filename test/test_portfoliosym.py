from __future__ import absolute_import
from unittest import TestCase
import unittest
from rewave.environment.long_portfolio import PortfolioSim, DELTA_MU
from datetime import date
import logging
from pprint import pprint

import numpy as np

class TestPortfolioSym(unittest.TestCase):
    def setUp(self):
        self.cost = 0.0025
        self.time_cost = 0
        self.batch_size = 50
        self.tickers_list = ['AAPL', 'MSFT', 'A']

        self.portfolio_sym = PortfolioSim(tickers_list=self.tickers_list, batch_size=self.batch_size, trading_cost=self.cost,time_cost=self.time_cost)



    def test_initialization(self):
        sym = self.portfolio_sym
        self.assertEqual([], sym.infos, 'infos set to empty')
        w0 = np.array([1.0] + [0.0] * len(self.tickers_list))
        self.assertEqual(w0.tolist(), sym.w0.tolist(), 'weight should be only for cash')
        self.assertEqual(1.0, sym.p0, 'portfolio value should start at 1.0')

    def test_calculate_after_commission(self):
        nb_tickers = len(self.tickers_list)
        w0 = np.array([1.0] + [0.0] * nb_tickers)
        w1 = np.concatenate(([0.0], np.ones(nb_tickers) / (nb_tickers)), axis=0)
        mu1 = self.portfolio_sym.calculate_pv_after_commission(w1, w0, self.cost)
        self.assertEqual(1.0-self.cost, mu1, 'cost should be equal to commission')
        w1 = np.ones(nb_tickers+1) / (nb_tickers+1)
        mu0 = 1.0 - self.cost * nb_tickers/(nb_tickers+1)
        mu1 = self.portfolio_sym.calculate_pv_after_commission(w1, w0, self.cost)
        self.assertLessEqual(DELTA_MU, abs(mu1-mu0), msg='cost should be equal to commission')

        w1 = w0
        mu1 = self.portfolio_sym.calculate_pv_after_commission(w1, w0, self.cost)
        self.assertAlmostEqual(1.0, mu1, msg='no commission if no change', delta=DELTA_MU)
        w1 = np.ones(nb_tickers + 1) / (nb_tickers + 1)
        w0 = w1
        mu1 = self.portfolio_sym.calculate_pv_after_commission(w1, w0, self.cost)
        self.assertAlmostEqual(1.0, mu1, msg='no commission if no change', delta=DELTA_MU)

    def test_step(self):
        nb_tickers = len(self.tickers_list)
        w0 = np.array([1.0] + [0.0] * nb_tickers)
        w1 = np.concatenate(([0.0], np.ones(nb_tickers) / (nb_tickers)), axis=0)
        y1 = np.concatenate(([1.0], 2.0* np.ones(nb_tickers)), axis=0)
        reward, info, done = self.portfolio_sym._step(w1, y1)
        self.assertEqual((1-self.cost)*2.0, self.portfolio_sym.p0, "all invested")
        self.assertEqual((1 - self.cost) * 2.0, info['portfolio_change'], "all invested")

        print("##################")
        print(self.portfolio_sym.p0)
        print(reward)
        pprint(info)

        y1 = np.ones(nb_tickers+1)
        reward, info, done = self.portfolio_sym._step(w1, y1)
        self.assertAlmostEqual((1 - self.cost) * 2.0, self.portfolio_sym.p0, msg="no change", delta=DELTA_MU)
        self.assertAlmostEqual(1.0, info['portfolio_change'], msg="no portfolio change", delta=DELTA_MU)

        print("##################")
        print(self.portfolio_sym.p0)
        print(reward)
        pprint(info)

        y1 = np.ones(nb_tickers + 1) * 0.5
        reward, info, done = self.portfolio_sym._step(w1, y1)
        self.assertAlmostEqual((1 - self.cost) * 1.0, self.portfolio_sym.p0, msg="no change", delta=DELTA_MU)
        self.assertAlmostEqual(0.5, info['portfolio_change'], msg="no portfolio change", delta=DELTA_MU)

        print("##################")
        print(self.portfolio_sym.p0)
        print(reward)
        pprint(info)

