from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

import logging
import sqlite3
from datetime import date, datetime

# from thewave.marketdata.tickerlist import TickerList
import numpy as np
import pandas as pd


DATABASE_DIR = path.realpath(__file__).\
    replace('rewave/marketdata/stooq.pyc','/marketdata/Stooq.db').\
    replace("rewave\\marketdata\\stooq.pyc","marketdata\\Stooq.db").\
    replace('rewave/marketdata/stooq.py','/marketdata/Stooq.db').\
    replace("rewave\\marketdata\\stooq.py","marketdata\\Stooq.db")

class StooqToDB:

    def __init__(self, root = "/", frequency='daily'):
        self.initialize_db()

        self.root = root

        self.frequency = frequency

    def initialize_db(self):
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            #cursor.execute('CREATE TABLE IF NOT EXISTS History (ticker varchar(20), date date, open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT, ex_dividend FLOAT,adj_open FLOAT, adj_high FLOAT, adj_low FLOAT, adj_close FLOAT, adj_volume FLOAT, PRIMARY KEY (ticker, date));')

            EXCHANGE_CREATE_STATEMENT = ('CREATE TABLE IF NOT EXISTS exchange (id INTEGER PRIMARY KEY AUTOINCREMENT,'
                                          'abbrev VARCHAR(32) NOT NULL,'
                                          'name VARCHAR(255) NOT NULL,'
                                          'city VARCHAR(255) NOT NULL,'
                                          'country VARCHAR(255) NOT NULL,'
                                          'currency VARCHAR(32) NOT NULL,'
                                          'timezone_offset time NOT NULL,'
                                          'created_date date_time NOT NULL,'
                                          'last_updated_date VARCHAR(255) NOT NULL);')

            DATA_VENDOR_CREATE_STATEMENT = ('CREATE TABLE IF NOT EXISTS data_vendor (id INTEGER PRIMARY KEY AUTOINCREMENT,'
                                         'name VARCHAR(64) NOT NULL,'
                                         'web_site_url VARCHAR(255) NOT NULL,'
                                         'support_email VARCHAR(255) NOT NULL,'
                                         'created_date date_time NOT NULL,'
                                         'last_updated_date VARCHAR(255) NOT NULL);')

            SYMBOL_CREATE_STATEMENT = ('CREATE TABLE IF NOT EXISTS symbol (id INTEGER PRIMARY KEY AUTOINCREMENT,'
                                        'exchange_id INTEGER,'
                                        'ticker VARCHAR(32) NOT NULL,'
                                        'instrument VARCHAR(64),'
                                        'name VARCHAR(255) NOT NULL,'
                                        'sector VARCHAR(255),'
                                        'currency varchar(32),'
                                        'created_date date_time NOT NULL,'
                                        'last_updated_date VARCHAR(255) NOT NULL,'
                                        'FOREIGN KEY(exchange_id) REFERENCES exchange(id) );')

            DAILY_PRICE_CREATE_STATEMENT = ('CREATE TABLE IF NOT EXISTS daily_price (id INTEGER PRIMARY KEY AUTOINCREMENT,'
                                           'data_vendor_id INTEGER,'
                                           'symbol_id INTEGER,'
                                           'price_date DATETIME NOT NULL,'
                                           'open FLOAT,'
                                           'high FLOAT,'
                                           'low FLOAT,'
                                           'close FLOAT,'
                                           'volume FLOAT,'
                                           'ex_dividend FLOAT,'
                                           'split_ratio FLOAT,'
                                           'adj_open FLOAT,'
                                           'adj_high FLOAT,'
                                           'adj_low FLOAT,'
                                           'adj_close FLOAT,'
                                           'adj_volume FLOAT,'
                                           'created_date date_time NOT NULL,'
                                           'last_updated_date VARCHAR(255) NOT NULL,'
                                           'FOREIGN KEY(data_vendor_id) REFERENCES data_vendor(id),'
                                           'FOREIGN KEY(symbol_id) REFERENCES symbol(id) );')


            cursor.execute(EXCHANGE_CREATE_STATEMENT)
            cursor.execute(DATA_VENDOR_CREATE_STATEMENT)
            cursor.execute(SYMBOL_CREATE_STATEMENT)
            cursor.execute(DAILY_PRICE_CREATE_STATEMENT)
            connection.commit()

    def set_ticker_id(self, ticker):
        connection = sqlite3.connect(DATABASE_DIR)

    def get_ticker_id(self, ticker):
        connection = sqlite3.connect(DATABASE_DIR)



    def update_ticker(self, ticker):
        connection = sqlite3.connect(DATABASE_DIR)
        logging.info("DB Update for ticker: %s " % (ticker))

        min_date = None
        max_date = None
        try:
            cursor = connection.cursor()
            min_date = cursor.execute('SELECT MIN(date) FROM daily_price WHERE ticker=?;', (ticker,)).fetchall()[0][0]
            max_date = cursor.execute('SELECT MAX(date) FROM daily_price WHERE ticker=?;', (ticker,)).fetchall()[0][0]
        finally:
            connection.commit()


        try:
            cursor = connection.cursor()
            #min_date = cursor.execute('SELECT MIN(date) FROM History WHERE ticker=?;', (ticker,)).fetchall()[0][0]
            #max_date = cursor.execute('SELECT MAX(date) FROM History WHERE ticker=?;', (ticker,)).fetchall()[0][0]

            if min_date==None or max_date==None:
                logging.info("DB REQUEST CREATION for ticker: %s " % (ticker))
                self.fill_ticker(ticker, cursor)
            else:
                now_ts = pd.Timestamp(datetime.now())
                max_ts = pd.Timestamp(max_date)
                if (max_date == None) or (now_ts - max_ts).days >1:
                    logging.info("DB REQUEST UPDATE for ticker: %s from: %s" % (ticker, max_date))
                    ticker_data = self._quandl_request.data(ticker, {'start_date': max_date})
                    for tick in ticker_data:
                        cursor.execute('INSERT OR IGNORE INTO History (date, ticker, open, high, low, close, '
                                       'volume, ex_dividend, '
                                       'adj_open, adj_high, adj_low, adj_close, adj_volume) '
                                       'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                       (tick[0], ticker, tick[1], tick[2], tick[3], tick[4],
                                        tick[5], tick[6],
                                        tick[8], tick[9], tick[10], tick[11], tick[12]))

            # if there is no data
        finally:
            connection.commit()
            connection.close()

if __name__ == "__main__":
    StooqToDB()