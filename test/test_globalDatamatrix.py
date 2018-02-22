from __future__ import absolute_import
from unittest import TestCase
import unittest
from rewave.marketdata.globaldatamatrix import HistoryManager
from rewave.tools.data import translate_to, translate_from, FEATURES_ADJUSTMENT, REVERTED_FEATURES_ADJUSTMENT

from datetime import date, datetime
import logging



class TestGlobalDataMatrix(unittest.TestCase):
    def setUp(self):
        self.start = date(2017, 1, 1)
        self.end = date(2018, 1, 1)
        self.features= ['close', 'high', 'low']
        self.tickers = ['AAPL','A']
        self.historymanager = HistoryManager(tickers = self.tickers)
        self.datamatrix = self.historymanager.get_global_panel(start=self.start,end=self.end, features=self.features, tickers = self.tickers)

    def test_count_features(self):
        self.assertEqual(self.datamatrix.shape[0], len(self.features))

    def test_count_tickers(self):
        self.assertEqual(self.datamatrix.shape[1], len(self.tickers))

    def test_count_periods(self):
        days = (self.end-self.start).days
        self.assertLessEqual(self.datamatrix.shape[2], days + 1)

    def test_historical_data(self):
        df = self.historymanager.historical_data(start=self.start,end=self.end, tickers = self.tickers,features=self.features)
        self.assertEquals(df.columns.levels[0].tolist(), self.tickers, "tickers not correctly set")
        self.assertEquals(df.columns.levels[1].tolist(), self.features, "features not correctly set")

    def test_adjustment(self):
        features_list = translate_to(source=self.features, dict=FEATURES_ADJUSTMENT)
        df = self.historymanager.historical_data(start=self.start, end=self.end, tickers=self.tickers,
                                                 features=self.features, adjusted=True)
        self.assertEquals(df.columns.levels[1].tolist(), self.features, "features not correctly set")

    #@TODO because of adj_close updated when big change, need to refresh ALL THE DB

if __name__ == '__main__':
    unittest.main()
