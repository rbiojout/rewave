#!/usr/bin/env python3
"""
Convenience wrapper for Interactive Brokers API.
"""

# Copyright (C) 2016  Doctor J
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import random
import sys
import threading
import time
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
import logging
from copy import copy
import math
from itertools import takewhile, tee, starmap
from queue import Queue, Empty
from typing import Optional, Tuple, Iterable, Union, Any, Callable
import unittest

"""
from ib.opt import ibConnection
from ib.ext.Contract import Contract
from ib.ext.Order import Order as IBOrder
from ib.ext.TickType import TickType
"""
from pytz import timezone, utc


__version__ = "0.3.1"
__all__ = ('IBroke', 'Instrument', 'Order', 'Bar', 'now')

#: Contract tuple type.  TODO: Want to be able to elide trailing values, I think.
ContractTuple = Tuple[str, str, str, str, str, float, str]
#: API warning codes that are not actually problems and should not be logged
BENIGN_ERRORS = (202, 2104, 2106, 2137)     # 202 is issued when you cancel an order.
#: API error codes indicating IB/TWS disconnection
DISCONNECT_ERRORS = (504, 502, 1100, 1300, 2110)
#: API error codes indicating reconnection (after a disconnect)
RECONNECT_CODES = (1102,)       # TODO: 1101 means you need to resubscribe to market data and account data.
#: When an order fails, the orderStatus message doesn't tell you why.  The description comes in a separate error message, so you gotta be able to tell if the "id" in an error message is an order id or a ticker id.
ORDER_RELATED_ERRORS = (103, 104, 105, 106, 107, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 129, 131, 132, 133, 134, 135, 136, 137, 140, 141, 144, 146, 147, 148, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 166, 167, 168, 201, 203, 303, 311, 312, 313, 314, 315, 325, 327, 328, 329, 335, 336, 337, 338, 339, 340, 341, 342, 343, 347, 348, 349, 350, 351, 352, 353, 355, 356, 358, 359, 360, 361, 362, 363, 364, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 382, 383, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 422, 423, 424, 425, 426, 427, 428, 429, 433, 434, 435, 436, 437, 512, 515, 516, 517, 10003, 10005, 10006, 10007, 10008, 10009, 10010, 10011, 10012, 10013, 10014, 10016, 10017, 10018, 10019, 10020, 10021, 10022, 10023, 10024, 10025, 10026, 10027, 10147)
#: Error codes related to data requests, i.e., the error id is a ticker id.
TICKER_RELATED_ERRORS = (101, 102, 138, 301, 300, 302, 309, 310, 316, 317, 321, 322, 354, 365, 366, 385, 386, 420, 510, 511, 519, 520, 524, 525, 529, 530,)
#: Errors requesting contract details
CONTRACT_REQUEST_ERRORS = (200,)
#: A commission value you'd never expect to see.  Sometimes we get bogus commission values.
CRAZY_HIGH_COMMISSION = 1000000
#: A fill profit you'd never expect to see.
CRAZY_HIGH_PROFIT = 1000000
#: Map verbosity levels to logger levels
LOG_LEVELS = {
    0: logging.CRITICAL,
    1: logging.ERROR,
    2: logging.WARNING,
    3: logging.INFO,
    4: logging.DEBUG,
    5: logging.DEBUG,
}
InstrumentDefaults = namedtuple('InstrumentDefaults', 'symbol sec_type exchange currency expiry strike opt_type')
#: Default values for instrument fields
INSTRUMENT_DEFAULTS = InstrumentDefaults(None, 'STK', 'SMART', 'USD', None, 0.0, None)


class Instrument:
    """Represents a stock, bond, future, forex currency pair, or option.

    Returned by :meth:`IBroke.get_instrument`, cannot be created directly by user code.
    """
    def __init__(self, broker, contract_details):
        """Create an Instrument object defining what will be traded, at which exchange and in which currency.

        :param IBroke broker: :class:`IBroke` instance
        :param ContractDetails contract_details: IBPy :class:`ContractDetails` object (must have valid `conID` from `contractDetails()`)
        """
        # TODO: The design of this class makes me uneasy.  Ideally it should be immutable, but the underlying ContractDetails can
        # change, e.g., market hours day-to-day.  And clients keep references around.
        if not contract_details or not contract_details.m_summary:
            raise ValueError('ContractDetails contained no Contract summary')
        self._broker = broker
        self._details = contract_details
        self._contract = self._details.m_summary
        if not self._contract.m_conId:
            raise ValueError('Contract must have conId (obtained from contractDetails()).')
        try:
            #: The leverage multiplier, i.e., what you multiply the quote by to get the actual underlying value."""
            self.leverage = float(self._contract.m_multiplier)      # Not a property method because we only want to parse / warn once.
        except (ValueError, TypeError):
            self._broker.log.debug("Error parsing contract ID {} multiplier '{}'; using leverage 1.0".format(self.id, self._contract.m_multiplier))
            self.leverage = 1.0
        tz = get_timezone(self._details.m_timeZoneId)
        #print('TRADING HOURS', self, '\n', self._details.m_tradingHours, '\nTIMEZONE', tz)
        self._trading_hours = self._normalize_trading_hours(self._parse_trading_hours(self._details.m_tradingHours), tz)
        self._liquid_hours = self._normalize_trading_hours(self._parse_trading_hours(self._details.m_liquidHours), tz)
        self._created_time = now().astimezone(tz)       # Used to know when our trading/liquid hours are stale

    @staticmethod
    def _parse_trading_hours(hours: str) -> Iterable[Tuple[datetime, datetime]]:
        """:Return: A tuple of pairs of naive (tz-unaware) :class:`datetime` objects giving the time ranges
        parsed from an IB trading hours string.

        Example:
            '20170621:1700-1515,1530-1600;20170622:1700-1515,1530-1600'
            '20170623:1715-1700;20170626:1715-1700'
        """
        for daystr in hours.split(';'):     # Regex can't handle varying number of repeated capture groups
            date, times = daystr.split(':')
            date = datetime.strptime(date, '%Y%m%d').date()
            if times != 'CLOSED':
                for hours in times.split(','):
                    start, end = hours.split('-')
                    start = datetime.strptime(start, '%H%M').time()
                    end = datetime.strptime(end, '%H%M').time()
                    yield datetime.combine(date, start), datetime.combine(date, end)

    @staticmethod
    def _normalize_trading_hours(datetimes: Iterable[Tuple[datetime, datetime]], tz: timezone) -> Tuple[Tuple[datetime, datetime], ...]:
        """:Return: a sorted tuple of :class`datetime` ranges (pairs) where "wraparound" time ranges have been replaced with
        properly ordered, collapsed ranges, and the given `timezone` has been set.

        Note this may change the number of time ranges.
        """
        if tz is None:
            raise ValueError('You better know what timezone your dates are in.')

        def normalize(start, end):
            assert start.date() == end.date()
            if start > end:     # When start > end, start is actually the day before
                start -= timedelta(days=1)
                assert start < end
            return tz.localize(start), tz.localize(end)     # Give them a timezone.  It is important to use pytz' localize() instead of creating a datetime with a tzinfo.

        normed = tuple(starmap(normalize, datetimes))
        assert all(e1 <= s2 for (_, e1), (s2, _) in pairwise(normed))       # End of last range is before (or eq) start of next range
        return normed

    @property
    def symbol(self):
        return self._contract.m_symbol

    @property
    def sec_type(self):
        return self._contract.m_secType

    @property
    def exchange(self):
        return self._contract.m_exchange

    @property
    def currency(self):
        return self._contract.m_currency

    @property
    def expiry(self):
        return self._contract.m_expiry

    @property
    def strike(self):
        return self._contract.m_strike

    @property
    def opt_type(self):
        return self._contract.m_right

    @property
    def id(self):
        """:Return: a unique ID for this instrument."""
        return IBroke._instrument_id_from_contract(self._contract)

    def tuple(self):
        """:Return: The instrument as a 7-tuple."""
        return tuple(getattr(self, prop) for prop in InstrumentDefaults._fields)

    def __str__(self):
        return str(self.tuple())

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """:Return: True iff `other` has the same IB Contract ID as this Instrument."""
        return self.id == other.id

    def __hash__(self):
        return self.id


class Order:
    """An order for an :class:`Instrument`.

    Not created by user code directly.
    """
    def __init__(self, id_, instrument, price, quantity, filled, open, cancelled):
        """
        :param int quantity: Positive for buy, negative for sell
        :param int filled: Number of shares filled.  NEGATIVE FOR SELL ORDERS (when `quantity` is negative).
          If `quantity` is -5 (sell short 5), then `filled` == -3 means 3 out of 5 shares have been sold.
        """
        self.id = id_
        self.instrument = instrument
        self.price = price
        self.quantity = quantity
        self.filled = filled
        self.avg_price = None
        self.open = open
        self.cancelled = cancelled
        self.profit = 0                 # Realized profit so far for this order, net commisssions (negative for loss).  Reflects IB's (strange) accounting.
        self.commission = 0
        self.open_time = None           # openOrder server time (epoch sec)
        self.fill_time = None           # Most recent fill (epoch sec)

    @property
    def complete(self):
        """:Return: True iff ``filled == quantity``."""
        return self.filled == self.quantity

    @staticmethod
    def _from_ib(order, order_id, instrument):
        """:Return: A new ibroke.Order object created from a :class:`ib.ext.Order.Order`."""
        qty = order.m_totalQuantity * (1 if order.m_action == 'BUY' else -1)
        return Order(order_id, instrument, price=order.m_lmtPrice or None, quantity=order.m_totalQuantity, filled=0, open=True, cancelled=False)

    def __repr__(self):
        return str(self)

    def __str__(self):
        inst = tuple(val for default, val in zip(INSTRUMENT_DEFAULTS, self.instrument.tuple()) if val != default)
        return "Order<{inst} {filled}/{quantity} @ {price} {open}{cancelled} #{id}>".format(
            id=self.id, inst=inst, filled=self.filled, quantity=self.quantity, price=self.price, open='open' if self.open else 'closed', cancelled=' cancelled' if self.cancelled else '')


Bar = namedtuple('Bar', ('time', 'bid', 'bidsize', 'ask', 'asksize', 'last', 'lastsize', 'lasttime', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'open_interest'))
Bar.__doc__ = """
Bar is the usual open / high / low / close trade prices you are probably familiar with from stock quotes,
plus the most recent bid, ask and trade prices, and the volume-weighted average trade price over the period.
Bar values, in order, are:

time
    The Unix timestamp of the end of the bar, in seconds since the epoch UTC as a float

bid
    The most recent price at which you can sell the instrument

bidsize
    The number of shares/contracts wanted at the bid price

ask
    The most recent price at which you can buy the instrument

asksize
    The number of shares/contracts available at the ask price

last
    The most recent price at which the instrument was traded

lastsize
    The number of shares/contracts exchanged in the most recent trade

lasttime
    The Unix timestamp of the last trade, in seconds since the epoch UTC as a float

open
    The last trade price prior to the start of this bar; usually always equal to the previous bar close

high
    The highest trade price during this bar (or the price of the most recent trade if none occurred during this bar)

low
    The lowest trade price during this bar (or the price of the most recent trade if none occurred during this bar)

close
    The last trade price during this bar (or the price of the most recent trade if none occurred during this bar)

vwap
    The `volume-weighted average price <https://en.wikipedia.org/wiki/Volume-weighted_average_price>`__
    of all trades **since the last bar**, or 0.0 if there were none.

volume
    The cumulative total number of shares/contracts traded today.  For US stocks, in lots, or shares divided by 100.

open_interest
    The number of oustanding contracts or options.  This is only available for a very small number of
    instruments; for most instruments it is always ``NaN``.
"""


class IBroke:
    """Interactive Brokers connection.

    It is not safe to call the methods of this object from multiple threads.
    """
    RTVOLUME = "233"
    RT_TRADE_VOLUME = "375"
    TICK_TYPE_RT_TRADE_VOLUME = 77

    def __init__(self, host='localhost', port=7497, client_id=None, timeout_sec=5, verbose=3):
        """Connect to Interactive Brokers.

        :param int client_id: An integer identifying which API client made an order.  In order to report
          and modify orders created before this connection, you must use the same `client_id` they were created with.
        :param float timeout_sec: If a connection cannot be established within this time, an exception is raised.  Also used internally for request timeouts.
        """
        super().__init__()
        client_id = client_id if client_id is not None else random.randint(1, 2**31 - 1)       # TODO: It might be nice if this was a consistent hash of the caller's __file__ or __module__ or something.
        self.log = create_logger(__name__, LOG_LEVELS[verbose])
        self.verbose = verbose
        self.account = None
        self.account_type = None                    # INDIVIDUAL for paper or real accounts, UNIVERSAL for demo
        self.__next_order_id = 0
        self._instruments = dict()                  # Maps instrument ID (contract ID) to Instrument object
        self._tick_errors = dict()                  # Maps instrument ID (contract ID) to queue of any exceptions (errors) generated by requesting that ticker
        self._tick_handlers = defaultdict(list)     # Maps instrument ID (contract ID) to list of functions to be called when that instrument's quote changes
        self._bar_handlers = defaultdict(list)      # Maps (bar_type, bar_size, instrument_id) to list of functions to be called with those bar events
        self._order_handlers = defaultdict(list)    # Maps instrument ID (contract ID) to list of functions to be called with order updates for that instrument
        self._alert_hanlders = defaultdict(list)    # Maps instrument ID (contract ID) to list of functions to be called with alerts for those tickers
        self._ticumulators = dict()                 # Maps instrument ID to Ticumulator for those ticks
        self._orders = dict()                       # Maps order_id to Order object
        self._executions = dict()                   # Maps execution IDs to order IDs.  Tracked because commissions are per-execution with no order ref.
        self._positions = dict()                    # Maps instrument ID to (number of shares held, average cost)
        self._reconcile_contract_requests = Queue() # Each _position() may generate a reqContractDetails request; it puts the req_id in this Queue; _positionEnd puts in a None and reconcile() waits on all of it.
        self._contract_details = []                 # Maps contractDetails() request id (int) to ContractDetails object.
        self._reconcile_open_orders_end = threading.Event() # Cleared and waited on by reconcile(), set by openOrderEnd
        self.timeout_sec = timeout_sec
        self.connected = None                       # Tri-state: None -> never been connected, False: initially was connected but not now, True: connected
        self._conn = ibConnection(host, port, client_id)
        self._conn.registerAll(self._handle_message)
        self._conn.connect()
        # The idea here is to catch errors synchronously, so if you can't connect, you know it at IBroke()
        start = time.time()
        while not self.connected:           # None initially meaning never connected; set False by _error(); set True by managedAccounts and nextValidId
            if self.connected is False or time.time() - start > timeout_sec:
                raise RuntimeError('Error connecting to IB')
            else:
                time.sleep(0.1)
        self.log.info('IBroke %s connected to %s:%d, client ID %d', __version__, host, port, client_id)
        self._conn.reqAccountSummary(0, 'All', 'AccountType')       # TODO: Wait, show value, verify
        time.sleep(0.25)
        self.reconcile()
        self.log_positions()
        self.log_open_orders()

    def get_instrument(self, symbol: Union[str, ContractTuple, int, Instrument], sec_type: str = 'STK', exchange: str = 'SMART', currency: str = 'USD', expiry: Optional[str] = None, strike: float = 0.0, opt_type: Optional[str] = None) -> Instrument:
        """Return an :class:`Instrument` object defining what will be purchased, at which exchange and in which currency.

            (symbol, sec_type, exchange, currency, expiry, strike, opt_type)

        :param symbol: The ticker symbol, IB contract tuple, IB contract ID, or :class:`Instrument`.
          Passing an :class:`Instrument` will always return the same object.
          Passing an (integer) contract ID will return an exisiting :class:`Instrument` if possible, otherwise create a new one.
        :param sec_type: The security type for the contract ('STK', 'FUT', 'CASH' (forex), 'OPT')
        :param currency: The currency in which to purchase the contract
        :param expiry: Future or option expiry date, YYYYMMDD.
        :param exchange: The exchange to trade the contract on.  Usually: stock: SMART, futures: GLOBEX, forex: IDEALPRO
        :param strike: The strike price for options
        :param opt_type: 'PUT' or 'CALL' for options
        """
        if isinstance(symbol, Instrument):
            return symbol
        elif isinstance(symbol, tuple):
            return self.get_instrument(*symbol)
        elif isinstance(symbol, int):
            contract = Contract()
            contract.m_conId = symbol
            inst = self._instruments.get(self._instrument_id_from_contract(contract))
            if inst is not None:
                return inst
        elif isinstance(symbol, str):
            contract = make_contract(symbol, sec_type, exchange, currency, expiry, strike, opt_type)
        else:
            raise ValueError("symbol must be string, int, tuple, or Instrument")

        # This functionality is split into request and response halves so elsewhere we can make the request in one callback and process the response in another.
        req_id = self._request_contract_details(contract)
        inst = self._handle_contract_details(req_id)
        earliest, latest = inst._trading_hours[0][0], inst._trading_hours[-1][1]
        self.log.debug('%s HOURS : %s -- %s  (%.0f h)', inst, earliest, latest, (latest - earliest).total_seconds() / 3600)
        return inst

    def _request_contract_details(self, contract):
        """Call reqContractDetails and stuff the results in ``self._contract_details[req_id]``, where `req_id` is the return value."""
        req_id = len(self._contract_details)  # TODO: race condition between getting length and extending
        self.log.debug('REQ CON DET %d ID %d', req_id, contract.m_conId)
        self._contract_details.append(Queue())  # Filled by _contractDetails(), capped off with None by _contractDetailsEnd()
        self._conn.reqContractDetails(req_id, contract)
        return req_id

    def _handle_contract_details(self, req_id):
        """Wait on `contractDetails` and `endContractDetails` for the given `req_id`.

        Creates a new Instrument and puts it in self._instruments.

        :raises ValueError: If waiting timed out or there was an exception otherwise.
        :return: Instrument created from the best-matching contract.
        """
        # Wait on all ContractDetails objects to fill in the queue, toss the terminating None, put into a tuple.
        details = tuple(takewhile(lambda v: v is not None, iter_except(lambda: self._contract_details[req_id].get(timeout=self.timeout_sec), Empty)))
        if not details:
            raise ValueError("Timed out looking for matching contracts for request ID".format(req_id))
        elif isinstance(details[0], Exception):  # Error
            raise details[0]
        best = choose_best_contract(details)
        self.log.debug('BEST %s', obj2dict(best.m_summary))
        inst = Instrument(self, best)
        self._instruments[inst.id] = inst
        self._positions.setdefault(inst.id, (0, None))  # ib.reqPositions() (called in reconcile()) only gives 0 positions for instruments traded recently, so we set our own
        return inst

    def register(self, instrument: Union[str, ContractTuple, int, Instrument], on_bar: Callable[[Instrument, Bar], None] = None, on_order: Callable[[Order], None] = None, on_alert: Callable[[Instrument, str], None] = None, bar_type: str = 'time', bar_size: float = 1.0) -> None:
        """Register bar, order, and alert handlers for an `instrument`.

        :param instrument: The instrument to register callbacks for.  Can be symbol, contract tuple, or :class:`Instrument`.
        :param on_bar: Call ``func(instrument, bar)`` with a :class:`Bar` every `bar_size` seconds.
        :param on_order: Call ``func(order)`` with an :class:`Order` object on order status changes for `instrument`.
        :param on_alert: Call ``func(instrument, alert_type)`` for notification of session start/end, disconnects/reconnects, trading halts, corporate actions, etc related to `instrument`.
        :param bar_type: The type of bar to generate: `'time'` to get periodic bars, or `'tick'` to get updates with every quote change.
        :param bar_size: The period of a bar in seconds.  Ignored for ``bar_type == 'tick'``.
        """
        assert bar_type in ('time', 'tick')
        assert bar_size > 0
        assert all(func is None or callable(func) for func in (on_bar, on_order, on_alert))
        assert not all(func is None for func in (on_bar, on_order, on_alert))
        instrument = self.get_instrument(instrument)
        if on_bar:
            # We need to accumulate ticks to make bars out of.
            if not self._tick_handlers.get(instrument.id):      # New ticker (get() does not insert into defaultdict)
                def unblock_register(*args):
                    """Temporary initial on_tick handler to unblock register() if a tick arrives"""
                    self._tick_errors[instrument.id].put_nowait(None)

                self._tick_errors[instrument.id] = Queue()      # _error() stuffs an exception in if it gets an error message; unblock_register stuffs None if it gets a tick
                self._tick_handlers[instrument.id].append(unblock_register)
                self._ticumulators[instrument.id] = Ticumulator()
                self._conn.reqMktData(instrument.id, instrument._contract, self.RTVOLUME, snapshot=False)       # Subscribe to continuous updates
                # TODO: Request an initial snapshot so we can start sending ticks without NaNs.
                # Hrm: Snapshots seem to take like 15 seconds...
                # self._conn.reqMktData(instrument.id, instrument._contract, None, snapshot=True)        # Request all fields once initially, so we don't have to wait for them to fill in
                # time.sleep(15)
                # Wait for errors
                # TODO: Something about waiting on errors.  There's no message on successful mkt data subscription,
                # so it's hard to know when it worked.  There won't always be a tick on success.
                # The delay compounds when subscribing to many tickers.  One shared
                # Queue to wait on?  Poll all queues (python doesn't have a wait-on-multiple-queues select() type thing)?
                try:
                    err = self._tick_errors[instrument.id].get(timeout=self.timeout_sec)
                    if err is not None:
                        raise err
                except Empty:       # No errors (or ticks) within timeout
                    pass
                assert len(self._tick_handlers[instrument.id]) == 1, 'Found more than initial tick handler on register {}: {}'.format(instrument.id, self._tick_handlers[instrument.id])
                self._tick_handlers[instrument.id].pop()        # Remove initial handler

            if bar_type == 'tick':
                self._tick_handlers[instrument.id].append(on_bar)
            elif bar_type == 'time':
                if len(frozenset(inst.id for _, _, inst in self._bar_handlers)) > 1:
                    raise NotImplementedError("Can't handle multiple bar types / sizes yet (instrument {})".format(instrument))
                self._bar_handlers[(bar_type, bar_size, instrument.id)].append(on_bar)
                RecurringTask(lambda: self._call_bar_handlers(bar_type, bar_size, instrument.id), interval_sec=bar_size, init_sec=1, daemon=True)        # This apparently sticks around even without maintaining a reference...

            self.log.debug('REGISTER %d %s', instrument.id, instrument)

        if on_order:
            self._order_handlers[instrument.id].append(on_order)
        if on_alert:
            self._alert_hanlders[instrument.id].append(on_alert)

    def order(self, instrument: Instrument, quantity: int, limit: float = 0.0, stop: float = 0.0, target: float = 0.0) -> Optional[Order]:
        """Place an order and return an Order object, or None if no order was made.

        The returned object does not change (will not update).
        """
        if target:
            raise NotImplementedError()
        if quantity == 0:
            return None
        if not self.connected:
            self.log.error('Cannot order when not connected')
            return None

        typemap = {
            (False, False): 'MKT',
            (False, True):  'LMT',
            (True, False):  'STP',
            (True, True):   'STP LMT',
        }

        # TODO: Check stop limit values are consistent
        order = IBOrder()
        order.m_action = 'BUY' if quantity >= 0 else 'SELL'
        order.m_totalQuantity = abs(quantity)
        order.m_orderType = typemap[(bool(stop), bool(limit))]
        order.m_lmtPrice = limit
        order.m_auxPrice = stop
        order.m_tif = 'DAY'     # Time in force: DAY, GTC, IOC, GTD
        order.m_allOrNone = False   # Fill or Kill
        order.m_goodTillDate = "" #  FORMAT: 20060505 08:00:00 {time zone}
        order.m_clientId = self._conn.clientId

        order_id = self._next_order_id()
        self.log.debug('ORDER %d: %s %s', order_id, obj2dict(instrument._contract), obj2dict(order))
        self._orders[order_id] = Order._from_ib(order, order_id, instrument)
        self.log.info('ORDER %s', self._orders[order_id])
        self._conn.placeOrder(order_id, instrument._contract, order)        # This needs to come after updating self._orders
        return copy(self._orders[order_id])

    def order_target(self, instrument, quantity, limit=0.0, stop=0.0):
        """Place orders as necessary to bring position in `instrument` to `quantity`.

        Bracket orders (with `target`) don't really make sense here.
        """
        return self.order(instrument, quantity - self.get_position(instrument), limit=limit, stop=stop)

    def get_position(self, instrument):
        """:Return: the number of shares of `instrument` held (negative for short)."""
        pos = self._positions.get(instrument.id)
        if pos is None:
            self.log.warning('get_position() for unknown instrument {}'.format(instrument))
            return 0
        return pos[0]

    def get_positions(self):
        """:Return: an iterator of ``(instrument, position, avg_cost)`` tuples for any non-zero positions in this account."""
        for inst_id, (pos, avg_cost) in self._positions.items():
            if pos:
                yield (self._instruments.get(inst_id), pos, avg_cost)

    def get_cost(self, instrument):
        """:Return: the average cost of currently held shares of `instrument`.  If no shares held, return None."""
        pos = self._positions.get(instrument.id)
        if pos is None:
            self.log.warning('get_cost() for unknown instrument {}'.format(instrument))
            return None
        return pos[1] or None

    def cancel(self, order):
        """Cancel an `order`."""
        self.log.info('CANCEL %s', order)
        if not self.connected:
            self.log.error('Cannot cancel order when disconnected')
        else:
            self._conn.cancelOrder(order.id)

    def cancel_all(self, instrument=None, hard_global_cancel=False):
        """Cancel all open orders.  If given, only cancel orders for `instrument`.

        :param bool hard_global_cancel: If True, issue a global cancel for ALL orders for this ENTIRE account,
          including orders made by other API clients and the TWS GUI.
        """
        # TODO: We might want to request all open orders, since our order status tracking might not be perfect.
        if hard_global_cancel:
            if instrument is not None:
                raise ValueError('instrument must be None for hard_global_cancel')
            self.log.info('GLOBAL CANCEL')
            self._conn.reqGlobalCancel()
        else:
            for order in self._orders.values():
                if order.open and (instrument is None or order.instrument == instrument):
                    self.cancel(order)

    def flatten(self, instrument=None, hard_global_cancel=False):
        """Cancel all open orders and set position to 0 for all instruments, or only for `instrument` if given.

        :param bool hard_global_cancel: If True, issue a global cancel for ALL orders for this ENTIRE account,
          including orders made by other API clients and the TWS GUI.
        """
        self.cancel_all(instrument, hard_global_cancel=hard_global_cancel)
        time.sleep(1)       # TODO: Maybe wait for everything to be cancelled.
        for inst in ((instrument,) if instrument else self._instruments.values()):
            self.order_target(inst, 0)

    def get_open_orders(self, instrument=None):
        """:Return: an iterable of all open orders, or only those for `instrument` if given."""
        for order in self._orders.values():
            if order.open and (instrument is None or order.instrument == instrument):
                yield copy(order)

    def reconcile(self):
        """Refresh the local state of orders and positions with those from the server.

        Note: Orders and positions can change while this method is executing.  To be sure
        the state is synced, cancel all orders and pause for a bit before calling this method.

        This method will retrieve all open orders for all API clients for this account.

        This method blocks until all orders and positions are synced.

        Updates the next order id.
        """
        self.log.debug('RECONCILE POSITIONS')
        if not self._reconcile_contract_requests.empty():
            strays = tuple(iter_except(self._reconcile_contract_requests.get_nowait, Empty))
            self.log.warning("reconcile() queue not empty: %s Attempt to call reconcile more than once concurrently?", strays)
        self._conn.reqPositions()               # generates _position() messages, which requests contract details and stuffs the request IDs into the _reconcile_contract_requests queue; positionEnd() stuffs a None
        # _position() itself will call _request_contract_details, and we wait on the results here.
        # Wait on all contractDetails request IDs to fill the queue, plus None from positionEnd; put into a tuple
        req_ids = tuple(iter_except(lambda: self._reconcile_contract_requests.get(timeout=self.timeout_sec), Empty))
        if not req_ids or req_ids[-1] is not None:
            self.log.warning("reconcile() timed out waiting for positions; positions may be stale")

        for req_id in req_ids:
            if req_id is not None:
                try:
                    self._handle_contract_details(req_id)
                except Exception as err:
                    self.log.error('In reconcile() for contract request %d: %s', req_id, str(err).replace('\n', ' '))

        # Get open orders second, since they may reference the instruments we just created above
        self.log.debug('RECONCILE ORDERS')
        self._reconcile_open_orders_end.clear()
        self._conn.reqAllOpenOrders()
        if not self._reconcile_open_orders_end.wait(timeout=self.timeout_sec):
            self.log.error('reconcile() timed out waiting for all open orders')

        self._conn.reqIds(-1)
        self.log.debug('RECONCILE END')

    def log_positions(self):
        """Log positions at INFO."""
        for inst, pos, cost in self.get_positions():
            self.log.info('POSITION %d @ %.2f %s', pos, cost, inst)

    def log_open_orders(self):
        """Log open orders at INFO."""
        for order in self.get_open_orders():
            self.log.info('OPEN ORDER %s', order)

    def market_open(self, instrument: Instrument, time: Optional[datetime] = None, afterhours: bool = True) -> bool:
        """:Return: True if `instrument` trades at the given `time`, which defaults to now.

        This only works for (roughly) the current and following day, and may not work for times in the past.

        :param time: Must be a timezone-aware datetime to compare to market hours.  Defaults to now.
        :param afterhours: If ``afterhours = False``, ``market_open()`` will only return True for times
          inside normal market hours.  If ``afterhours = True``, it will return true for times inside
          afterhours trading as well.  (Technically ``afterhours = False`` means only return true during
          "liquid" market hours, according to IB.)
        :raises ValueError: If `time` is outside the known schedule for this instrument.  Usually that's only
          today and tomorrow.
        """
        # This is an IBroke method instead of Instrument to avoid mutuable Instrument state.
        # The info underlying an Instrument can change (e.g. market hours), but we want the canonical Instrument with the latest info.
        # So we always look it up from IBroke, refreshing if necessary.
        now_ = now()
        if time is None:
            time = now_
        if not time.tzinfo:
            raise ValueError('Time must have a timezone.')

        instrument = self._ensure_fresh_instrument_data(instrument)
        open_hours = instrument._trading_hours if afterhours else instrument._liquid_hours
        assert open_hours, 'Empty trading hours'
        earliest, latest = open_hours[0][0], open_hours[-1][1]
        if time < earliest and time < now_ - timedelta(seconds=self.timeout_sec):
            raise ValueError('Time {} earlier than available schedule {}'.format(time, earliest))
        if time > latest:
            self.log.warning('market_open() request {} beyond time horizon {}, assuming closed.'.format(time, latest))
        return any(start <= time <= end for start, end in open_hours)

    def market_hours(self, instrument: Instrument, afterhours: bool = True) -> Tuple[Optional[datetime], Optional[datetime]]:
        """:Return: the next market opening and closing time of the given `instrument`.

        Note either may be sooner than the other, and either or both may be None.

        :param afterhours: If True, return next times for after hours trading.
          If False, return next times for regular trading hours.
        """
        now_ = now()
        instrument = self._ensure_fresh_instrument_data(instrument)      # Update market hours if necessary
        open_hours = instrument._trading_hours if afterhours else instrument._liquid_hours
        latest = open_hours[-1][1] if open_hours else None
        if not latest or now_ > latest:
            self.log.warning('market_hours() request {} beyond time horizon {}, no hours available.'.format(now_, latest))
        open_, close = None, None
        for start, end in open_hours:
            if open_ is None and now_ <= start:
                open_ = start
            if close is None and now_ <= end:
                close = end
        return open_, close

    def _ensure_fresh_instrument_data(self, instrument: Instrument) -> Instrument:
        """Check if market hours data for `instrument` is stale and re-request, returning a new Instrument."""
        instrument = self.get_instrument(instrument.id)  # Lookup canonical Instrument by id; passing an Instrument returns the same object.
        # If market hours data is out of date, refresh.
        # Since we only get data for today and tomorrow, and not data for closed days, refresh if today != instrument timestamp day
        # (Hard to do just based on our market hours data structure, without timestamp, since entire closed days aren't represented.)
        today = now().astimezone(instrument._created_time.tzinfo).date()
        if instrument._created_time.date() != today:
            earliest, latest = instrument._trading_hours[0][0], instrument._trading_hours[-1][1]
            self.log.debug('REFRESH {} before: {} -- {}  ({:.0f} h)'.format(instrument, earliest, latest, (latest - earliest).total_seconds() / 3600))
            req_id = self._request_contract_details(instrument._contract)
            instrument = self._handle_contract_details(req_id)  # TODO: Waiting might lead to deadlock if we're called in another request?
            earliest, latest = instrument._trading_hours[0][0], instrument._trading_hours[-1][1]
            self.log.debug('REFRESH {} after : {} -- {}  ({:.0f} h)'.format(instrument, earliest, latest, (latest - earliest).total_seconds() / 3600))
        return instrument

    def disconnect(self):
        """Disconnect from IB, rendering this object mostly useless."""
        self.connected = False
        self._conn.disconnect()

    def _next_order_id(self):
        """Increment the internal order id counter and return it."""
        self.__next_order_id += 1
        return self.__next_order_id

    def _call_order_handlers(self, order):
        """Call any order handlers registered for ``order.instrument``."""
        for handler in self._order_handlers.get(order.instrument.id, ()):
            handler(copy(order))

    def _call_tick_handlers(self, ticker_id, tick):
        """Call any tick handlers for the given `ticker_id` with the given `tick` tuple."""
        tick = Bar._make(tick)
        instrument = self._instruments.get(ticker_id)
        if instrument is None:
            self.log.warning('No instrument found for ID %d calling tick handlers', ticker_id)
        else:
            for handler in self._tick_handlers.get(ticker_id, ()):      # get() does not insert into the defaultdict
                handler(instrument, tick)

    def _call_alert_handlers(self, alert, ticker_id=None):
        """Call all alert handlers with the given `alert`, or only those registered for a given `ticker_id` if given."""
        if ticker_id is None:
            for ticker_id in self._alert_hanlders:
                self._call_alert_handlers(alert, ticker_id)     # Oooh, recursion
        else:
            instrument = self._instruments.get(ticker_id)
            if instrument is None:
                self.log.warning('No instrument found for ID %d calling alert handlers', ticker_id)
            else:
                for handler in self._alert_hanlders.get(ticker_id, ()):      # get() does not insert into the defaultdict
                    handler(instrument, alert)

    def _call_bar_handlers(self, bar_type, bar_size, ticker_id):
        """Generate a bar (of the given `bar_type` and `bar_size`) for `ticker_id` and call any registered bar handlers."""
        instrument = self._instruments.get(ticker_id)
        acc = self._ticumulators.get(ticker_id)
        handlers = self._bar_handlers.get((bar_type, bar_size, ticker_id))
        if acc is None or instrument is None or handlers is None:
            self.log.warning('No instrument, ticumulator, or handlers found for ID %d calling %s %f bar handlers', ticker_id, bar_type, bar_size)
        else:
            bar = Bar._make(acc.bar())
            for handler in handlers:
                handler(instrument, bar)

    @staticmethod
    def _instrument_id_from_contract(contract):
        if not contract.m_conId:        # 0 is default
            raise ValueError('Invalid contract ID {} for contract {}'.format(contract.m_conId, obj2dict(contract)))
        return contract.m_conId


    ###########################################################################
    # Message Handlers
    ###########################################################################

    def _handle_message(self, msg):
        """Root message handler, dispatches to methods named `_typeName`.

        E.g., `tickString` messages are dispatched to `self._tickString()`.
        """
        if self.verbose >= 5:
            self.log.debug('MSG %s', str(msg))

        name = getattr(msg, 'typeName', None)
        if not name or not name.isidentifier():
            self.log.error('Invalid message name %s', name)
            return
        handler = getattr(self, '_' + name, self._defaultHandler)
        if not callable(handler):
            self.log.error("Message handler '%s' (type %s) is not callable", str(handler), type(handler))
            return
        handler(msg)

    def _error(self, msg):
        """Handle error messages from the API."""
        code = getattr(msg, 'errorCode', None)
        if not isinstance(code, int):
            self.log.error(str(msg).replace('\n', ' '))
        elif code in BENIGN_ERRORS:
            pass
        elif code in DISCONNECT_ERRORS:
            self.log.error(msg.errorMsg.replace('\n', ' ') + ' [{}]'.format(msg.errorCode))
            self.connected = False
            self._call_alert_handlers('Disconnect')
        elif code in RECONNECT_CODES:
            # I originally thought receipt of any non-error message meant we must be (re-)connected, but it turns out market data (tickXXX messages)
            # are separate connections from commands, and you can be connected to none, either, or both.  self.connected means the command
            # connection.  So we need to wait specifically for a reconnect message (and not just any message) to set self.connected.
            self.connected = True
            self._call_alert_handlers('Reconnect')
        elif 2100 <= code < 2200:
            self.log.warning(msg.errorMsg.replace('\n', ' ') + ' [{}]'.format(msg.errorCode))
        else:
            if code in ORDER_RELATED_ERRORS:         # TODO: Some of these are actually warnings (like 399, sometimes...?)...
                order = self._orders.get(msg.id)
                errorMsg = msg.errorMsg.replace('\n', ' ')
                self.log.error('ORDER ERR %s %s [%d]', order, errorMsg, code)
                if order:
                    order.cancelled = True
                    order.open = False
                    order.message = errorMsg
                    # TODO: This "error" changes the price but does not cancel (not sure of code): "Order Message:\nSELL 2 ES DEC'16\nWarning: Your order was repriced so as not to cross a related resting order"
                    self._call_order_handlers(order)

            elif code in TICKER_RELATED_ERRORS:
                self.log.error(str(msg).replace('\n', ' '))
                err_q = self._tick_errors.get(msg.id)       # register() puts a Queue here and waits for any errors (with timeout)
                if err_q is None:
                    self.log.warning('Got ticker error for unexpected request id {}: {} [{}]'.format(msg.id, msg.errorMsg.replace('\n', ' '), code))
                else:
                    err_q.put_nowait(ValueError('{} [{}]'.format(msg.errorMsg.replace('\n', ' '), code)))

            elif code in CONTRACT_REQUEST_ERRORS:
                self.log.error(str(msg).replace('\n', ' '))
                if msg.id >= len(self._contract_details):
                    self.log.error('No request slot for contract details {} found'.format(msg.id))
                else:
                    self._contract_details[msg.id].put_nowait(ValueError(msg.errorMsg))
                    self._contract_details[msg.id].put_nowait(None)     # None signals end of messages in queue

            else:
                self.log.error(msg.errorMsg.replace('\n', ' ') + ' [{}]'.format(msg.errorCode))

    def _managedAccounts(self, msg):
        """Save the account number."""
        self.connected = True
        accts = msg.accountsList.split(',')
        if len(accts) != 1:
            raise ValueError('Multiple accounts not supported.  Accounts: {}'.format(accts))
        self.account = accts[0]
        if self.account and self.account_type:
            self.log.info('Account %s type %s', self.account, self.account_type)

    def _accountSummary(self, msg):
        """Save the account type."""
        if msg.tag == 'AccountType':
            self.account_type = msg.value
        if self.account and self.account_type:
            self.log.info('Account %s type %s', self.account, self.account_type)

    def _tickSize(self, msg):
        """Called when market data tick sizes change."""
        acc = self._ticumulators.get(msg.tickerId)
        if acc is None:
            self.log.warning('No Ticumulator found for ticker id %d', msg.tickerId)
            return
        if msg.field == TickType.BID_SIZE:
            acc.add('bidsize', msg.size)
        elif msg.field == TickType.ASK_SIZE:
            acc.add('asksize', msg.size)
        elif msg.field == TickType.LAST_SIZE:
            pass    # RTVOLUME is faster, more accurate, and doesn't have dupes.
            # acc.add('lastsize', msg.size)
        elif msg.field == TickType.VOLUME:
            pass
            # VOLUME only prints rarely, can be inaccurate, and differs widely from RTVOLUME.  Prefer RTVOLUME instead.
            # if math.isnan(acc.volume):
            #    acc.add('volume', msg.size)
        elif msg.field == TickType.OPEN_INTEREST:
            acc.add('open_interest', msg.size)

        self._call_tick_handlers(msg.tickerId, acc.peek())

    def _tickPrice(self, msg):
        """Called when market data tick prices change."""
        acc = self._ticumulators.get(msg.tickerId)
        if acc is None:
            self.log.warning('No Ticumulator found for ticker id %d', msg.tickerId)
            return
        if msg.field == TickType.BID:
            if msg.price >= 0.0:        # IB sometimes returns -1
                acc.add('bid', msg.price)
        elif msg.field == TickType.ASK:
            if msg.price >= 0.0:        # IB sometimes returns -1
                acc.add('ask', msg.price)
        elif msg.field == TickType.LAST:
            pass    # RTVOLUME is faster, more accurate, and doesn't have dupes.
            # acc.add('last', msg.price)

        self._call_tick_handlers(msg.tickerId, acc.peek())

    def _tickString(self, msg):
        """Called for real-time volume ticks and last trade times."""
        acc = self._ticumulators.get(msg.tickerId)
        if acc is None:
            self.log.warning('No Ticumulator found for ticker id %d', msg.tickerId)
            return

        if msg.tickType == TickType.LAST_TIMESTAMP:
            pass # RTVOLUME is faster, more accurate, and doesn't have dupes.
            # acc.add('lasttime', int(msg.value))
        elif msg.tickType == TickType.RT_VOLUME:    # or msg.tickType == self.TICK_TYPE_RT_TRADE_VOLUME:       # RT Trade Volume still in beta I guess
            # semicolon-separated string of:
            # last trade price ; last trade size ; last trade time in epoch ms; total volume for the day (in lots (of 100 for stocks)) ; VWAP for the day ; single trade flag (True indicates the trade was filled by a single market maker; False indicates multiple market-makers helped fill the trade)
            vals = msg.value.split(';')
            if vals[0]:     # Sometimes price is the empty string (odd lots?); in this case size == 0 and volume doesn't change, so we skip it.  (I think this is what RT Trade Volume is supposed to fix?)
                try:
                    lastprice, lastsize, lasttime, volume, vwap = map(float, vals[:5])
                except ValueError as err:
                    self.log.warning("Error parsing RTVOLUME tickString '%s': %s", msg.value, str(err))
                    return
                acc.add('last', lastprice)
                acc.add('lastsize', lastsize)       # Ticumulator likes lastsize to come after last
                acc.add('lasttime', lasttime / 1000)
                acc.add('volume', volume)
        else:       # Unknown tickType
            return

        self._call_tick_handlers(msg.tickerId, acc.peek())

    def _tickGeneric(self, msg):
        """Called for trading halts."""
        if msg.tickType == TickType.HALTED:
            if msg.value == 0:
                self._call_alert_handlers('Unhalt', msg.tickerId)
            else:
                self._call_alert_handlers('Halt', msg.tickerId)  # TODO: Alert enum or something

    def _nextValidId(self, msg):
        """Sets next valid order ID."""
        self.connected = True
        if msg.orderId >= self.__next_order_id:
            self.__next_order_id = msg.orderId
        else:
            self.log.warning('nextValidId {} less than current id {}'.format(msg.orderId, self.__next_order_id))

    def _contractDetails(self, msg):
        """Callback for reqContractDetails.  Called multiple times with all possible matches for one request,
        followed by a contractDetailsEnd.  We put the responses in to a dict of Queues indexed by request id (self._contract_details[req_id]),
        followed by None to indicate the end."""
        self.log.debug('DETAILS %d ID %d %s', msg.reqId, msg.contractDetails.m_summary.m_conId, obj2dict(msg.contractDetails))
        if msg.reqId >= len(self._contract_details):
            self.log.error('Could not find contract details slot %d for %s', msg.reqId, obj2dict(msg.contractDetails))
        else:
            self._contract_details[msg.reqId].put_nowait(msg.contractDetails)

    def _contractDetailsEnd(self, msg):
        """Called after all contractDetails messages for a given request have been sent.  Stuffs None into the Queue
        for the request ID to indicate the end."""
        self.log.debug('DETAILS END %s', msg)
        if msg.reqId >= len(self._contract_details):
            self.log.error('Could not find contract details slot %d for %s', msg.reqId, obj2dict(msg.contractDetails))
        else:
            self._contract_details[msg.reqId].put_nowait(None)

    def _orderStatus(self, msg):
        """Called with changes in order status.

        Except:
        "Typically there are duplicate orderStatus messages with the same information..."
        "There are not guaranteed to be orderStatus callbacks for every change in order status."

        orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice
        """
        order = self._orders.get(msg.orderId)
        if not order:
            self.log.error('Got orderStatus for unknown orderId {}'.format(msg.orderId))
            return

        # TODO: Worth making these immutable and replacing them?  Or *really* immutable and appending to a list of them?
        if order.open_time is None:
            order.open_time = time.time()
        if msg.status in ('ApiCanceled', 'Cancelled'):       # Inactive can mean error or not.  And yes, they appear to spell cancelled differently.
            if not order.cancelled:     # Only log the first time (can be dupes)
                self.log.info('CANCELLED %s', order)
            order.cancelled = True
            order.open = False
        elif msg.filled > abs(order.filled):      # Suppress duplicate / out-of-order fills  (order.filled is negative for sells)
            order.filled = int(math.copysign(msg.filled, order.quantity))
            order.avg_price = msg.avgFillPrice
            if order.filled == order.quantity:
                order.open = False
            self._call_order_handlers(order)

    def _openOrder(self, msg):
        """Called when orders are submitted and completed.

        Fields:
        orderId
        contract
        order: m_action, m_orderType, m_totalQuantity, m_clientId, m_tif, m_auxPrice, m_lmtPrice, ...
        orderState: m_status, m_commission, m_warningText, ...
        """
        self.log.debug('STATE %d %s', msg.orderId, obj2dict(msg.orderState))
        order = self._orders.get(msg.orderId)
        if not order:
            self.log.info('EXOGENOUS ORDER #%d for %s', msg.orderId, instrument_tuple_from_contract(msg.contract))
            instrument = self._instruments.get(msg.contract.m_conId)
            if instrument is None:
                self.log.error('Open order #%d for unknown instrument %s', msg.orderId, instrument_tuple_from_contract(msg.contract))
                return
            else:
                order = self._orders[msg.orderId] = Order._from_ib(msg.order, msg.orderId, instrument)

        assert order.id == msg.orderId
        assert order.instrument._contract.m_symbol == msg.contract.m_symbol     # TODO: More thorough equality
        if order.open_time is None:
            order.open_time = time.time()
        # possible status: Submitted Cancelled Filled Inactive
        if msg.orderState.m_status == 'Cancelled':
            order.cancelled = True
            order.open = False
        elif msg.orderState.m_status == 'Filled':       # Filled means completely filled
            if order.open:      # Only log first of dupe msgs
                self.log.info('COMPLETE %s avg price %f', order, order.avg_price)
            order.open = False

        if msg.orderState.m_warningText:
            warnText = msg.orderState.m_warningText.replace('\n', ' ')
            self.log.warning('Order %d: %s', msg.orderId, warnText)
            order.message = warnText

    def _openOrderEnd(self, msg):
        """Called after reqOpenOrders or reqAllOpenOrders to indicate all open orders have been sent."""
        self._reconcile_open_orders_end.set()

    def _execDetails(self, msg):
        """Called on order executions."""
        order = self._orders.get(msg.execution.m_orderId)
        if not order:
            self.log.error('Got execDetails for unknown orderId {}'.format(msg.execution.m_orderId))
            return
        exec = msg.execution
        # TODO: 5 digits of precision on price for forex
        self.log.info('EXEC %(symbol)s %(qty)d @ %(price).2f (%(total_qty)d filled) order %(id)d pos %(pos)d' % dict(time=exec.m_time, id=order.id, symbol=order.instrument.symbol, qty=int(math.copysign(exec.m_shares, order.quantity)), price=exec.m_price, total_qty=int(math.copysign(exec.m_cumQty, order.quantity)), pos=self.get_position(order.instrument)))
        assert order.id == exec.m_orderId
        if order.open_time is None:
            order.open_time = time.time()
        self._executions[exec.m_execId] = order.id      # Track which order executions belong to, since commissions are per-exec
        if exec.m_cumQty > abs(order.filled):           # Suppress duplicate / late fills.  Remember, kids: sells are negative!
            # TODO: Save server time delta
            order.fill_time = time.time()
            order.filled = int(math.copysign(exec.m_cumQty, order.quantity))
            order.avg_price = exec.m_avgPrice
            if order.filled == order.quantity:
                order.open = False
            # Call order handlers in commissionReport() instead of here so we can include commission info.

    def _position(self, msg):
        """Called when positions change; gives new position.

        If the instrument is unknown (not in self.instruments[]), we assume it's from a reconcile() call,
        make a contract details request, and stuff the request ID in _reconcile_contract_requests.
        """
        inst_id = self._instrument_id_from_contract(msg.contract)
        self.log.debug('POS %d %s %s', msg.pos, inst_id, obj2dict(msg.contract))
        # So: it's possible we don't have this Instrument in self._instruments, since the account may have had open positions before we started.
        # However, we can't wait for the look up (contractDetails) here because it causes a deadlock in IBPy (this method is an IBPy callback, so no other callbacks will fire until it returns).
        # So, we put the reqContractDetails request ID in a Queue, then wait (for all the contractDetailsEnds) in reconcile() and create the Instruments there.
        if inst_id not in self._instruments:
            self.log.debug('POS INST ID %d not found: %s', inst_id, self._instruments.keys())
            self._reconcile_contract_requests.put_nowait(self._request_contract_details(msg.contract))
        try:
            multiplier = float(msg.contract.m_multiplier)
        except (ValueError, TypeError):
            # We don't warn because this is called all the time.
            multiplier = 1.0

        self._positions[inst_id] = (msg.pos, msg.avgCost / multiplier)

    def _positionEnd(self, msg):
        """Called when all positions have been sent after a call to reqPositions; signals `reconcile()` the positions have been received."""
        self.log.debug('POSITION END')
        self._reconcile_contract_requests.put_nowait(None)       # Signal reconcile() we got all positions

    def _commissionReport(self, msg):
        """Called after executions; gives commission charge and PNL.  Calls order handlers."""
        # In theory we might be able to use orderState instead of commissionReport, but...
        # It's kinda whack.  Sometime's it's giant numbers, and there are dupes so it's hard to use.
        # TODO: We might want to guard against duplicate commissionReport messages.  Not sure if they happen or not.  But since we do accounting here...
        report = msg.commissionReport
        self.log.debug('COMM %s', vars(report))
        order = self._orders.get(self._executions.get(report.m_execId))
        if order:
            if 0 <= report.m_commission < CRAZY_HIGH_COMMISSION:        # We sometimes get bogus placeholder values
                order.commission += report.m_commission
            if -CRAZY_HIGH_PROFIT < report.m_realizedPNL < CRAZY_HIGH_PROFIT:
                order.profit += report.m_realizedPNL
            # TODO: We're potentially calling handlers more than once, here and in orderStatus
            # TODO: register() flag to say only fire on_order() events on totally filled, or cancel/error.
            self._call_order_handlers(order)
        else:
            self.log.error('No order found for execution {}'.format(report.m_execId))

    def _connectionClosed(self, msg):
        """Called when TWS straight drops yo shizzle."""
        self.connected = False
        self._call_alert_handlers('Connection Closed')

    def _defaultHandler(self, msg):
        """Called when there is no other message handler for `msg`."""
        if self.verbose < 5:        # Don't log again if already logged in main handler
            self.log.debug('MSG %s', msg)


class Ticumulator:
    """Accumulates ticks (bid/ask/last/volume changes) into bars (open/high/low/close/vwap).

    Bars contains the traditional OHLCV data, as well as bid/ask/last data and volume-weighted average price.
    You can use the :class:`Bar` namedtuple to wrap the output of this class for convenient attribute access.

    `bar()` will return data since the last `bar()` call (or creation), allowing you to make bars of any duration you like.

    Until a tick of each type has been added, the first results may contain ``NaN`` values and the volume may be
    off.

    `time` is Unix timestamp (float sec since epoch) of the end of the bar; `lasttime` is Unix time of last trade.
    `volume` is total cumulative volume for the day.  For US stocks, it is divided by 100.
    """
    #: 'what' inputs to `add()`
    INPUT_FIELDS = ('time', 'bid', 'bidsize', 'ask', 'asksize', 'last', 'lastsize', 'lasttime', 'volume', 'open_interest')

    def __init__(self):
        # Input
        self.time = float('NaN')
        self.bid = float('NaN')
        self.bidsize = float('NaN')
        self.ask = float('NaN')
        self.asksize = float('NaN')
        self.last = float('NaN')
        self.lastsize = float('NaN')
        self.lasttime = float('NaN')
        self.volume = float('NaN')
        self.open_interest = float('NaN')
        # Computed for bar
        self.open = float('NaN')
        self.high = float('NaN')
        self.low = float('NaN')
        self.close = float('NaN')
        self.sum_last = 0.0     # For VWAP
        self.sum_vol = 0.0

    def add(self, what, value):
        """Update this Ticumulator with an input type ``what`` with the given float ``value``.

        Valid ``what`` values are the :attribute:`INPUT_FIELDS` (except `time`).
        """
        self.time = time.time()
        if what not in self.INPUT_FIELDS[1:]:
            raise ValueError("Invalid `what` '{}'".format(what))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid value {}".format(value))

        setattr(self, what, value)
        if what == 'last':      # OHLC prices are trade prices
            if math.isnan(self.open):      # Very first datapoint ever
                self.open = self.high = self.low = self.close = value
            self.high = max(self.high, value)
            self.low = min(self.low, value)
            self.close = value

        # For vwap.  We arrange that lastsize comes in after the corresponding last price, since we get both from RTVOLUME.
        if what == 'lastsize':
            self.sum_last += self.last * self.lastsize      # self.lastsize == value, having been set above
            self.sum_vol += self.lastsize

    @property
    def vwap(self):
        return (self.sum_last / self.sum_vol) if self.sum_vol else 0.0

    def bar(self):
        """:Return: a tuple of `(time, bid, bidsize, ask, asksize, last, lastsize, lasttime, open, high, low, close, vwap, volume, open_interest)`.

        Reset the OHLC accumulators for the next bar.  OHLC values are for last trade prices only (not bid and ask).
        `time` is effectively the close (bar end) time.  `volume` is cumulative daily.

        .. seealso:: :class:`Bar`
        """
        bar = self.peek()
        self.open = self.close
        self.high = self.last
        self.low = self.last
        self.sum_last = self.sum_vol = 0.0
        return bar

    def peek(self):
        """:Return: a tuple of `(time, bid, bidsize, ask, asksize, last, lastsize, lasttime, open, high, low, close, vwap, volume, open_interest)`.

        Does not affect accumulators.

        .. seealso:: :class:`Bar`
        """
        return time.time(), self.bid, self.bidsize, self.ask, self.asksize, self.last, self.lastsize, self.lasttime, self.open, self.high, self.low, self.close, self.vwap, self.volume, self.open_interest


class RecurringTask(threading.Thread):
    """Calls a function at a sepecified interval."""
    def __init__(self, func, interval_sec, init_sec=0, *args, **kwargs):
        """Call `func` every `interval_sec` seconds.

        Starts the timer. Accounts for the runtime of `func` to make intervals as close to `interval_sec` as possible.
        args and kwargs are passed to `Thread()`.

        :param func func: Function to call
        :param float interval_sec: Call `func` every `interval_sec` seconds
        :param float init_sec: Wait this many seconds initially before the first call
        """
        super().__init__(*args, **kwargs)
        assert interval_sec > 0
        self._func = func
        self.interval_sec = interval_sec
        self.init_sec = init_sec
        self._running = True
        self._functime = None       # Time the next call should be made
        self.start()

    def __repr__(self):
        return 'RecurringTask({}, {}, {})'.format(self._func, self.interval_sec, self.init_sec)

    def run(self):
        """Start the recurring task."""
        if self.init_sec:
            time.sleep(self.init_sec)
        self._functime = time.time()
        while self._running:
            start = time.time()
            self._func()
            self._functime += self.interval_sec
            if self._functime - start > 0:
                time.sleep(self._functime - start)

    def stop(self):
        """Stop the recurring task."""
        self._running = False


def now() -> datetime:
    """:Return: the current time in UTC, with timezone."""
    return datetime.utcnow().replace(tzinfo=utc)


def make_contract(symbol, sec_type='STK', exchange='SMART', currency='USD', expiry=None, strike=0.0, opt_type=None):
    """:Return: an (unvalidated, no conID) IB Contract object with the given parameters."""
    contract = Contract()
    contract.m_symbol = symbol
    contract.m_secType = sec_type
    contract.m_exchange = exchange
    contract.m_currency = currency
    contract.m_expiry = expiry
    contract.m_strike = strike
    contract.m_right = opt_type
    return contract


def instrument_tuple_from_contract(contract):
    """:Return: an instrument tuple created from the fields of a Contract object."""
    return contract.m_symbol, contract.m_secType, contract.m_exchange, contract.m_currency, contract.m_expiry, contract.m_strike, contract.m_right


def choose_best_contract(details):
    """:Return: the "best" contract from the list of ``ContractDetails`` objects `details`, or None
    if there is no unambiguous best."""
    if not details:
        return None
    elif len(details) == 1:
        return details[0]

    types = frozenset(det.m_summary.m_secType for det in details)
    if len(types) == 1 and 'FUT' in types:      # Futures: choose nearest expiry
        best = min(details, key=lambda det: det.m_contractMonth)
    else:
        # TODO: Stocks, options, forex?
        return None
    return best


def obj2dict(obj):
    """Convert an (IBPy) object to a dict containing any fields with non-default values."""
    default = obj.__class__()
    return {field: val for field, val in vars(obj).items() if val != getattr(default, field, None)}


def get_timezone(abbrev: str) -> timezone:
    """:Return: a pytz :class:`timezone` object for a given IB abbreviation."""
    #: Maps timezone abbreviations returned in ContractDetails objects (from Java?) to "standard" tz names
    #: From http://grepcode.com/file/repository.grepcode.com/java/root/jdk/openjdk/8u40-b25/sun/util/calendar/ZoneInfoFile.java/#219
    TIMEZONE_ABBREVS = {
        "ACT": "Australia/Darwin",
        "AET": "Australia/Sydney",
        "AGT": "America/Argentina/Buenos_Aires",
        "ART": "Africa/Cairo",
        "AST": "America/Anchorage",
        "BET": "America/Sao_Paulo",
        "BST": "Asia/Dhaka",
        "CAT": "Africa/Harare",
        "CNT": "America/St_Johns",
        "CST": "America/Chicago",
        "CTT": "Asia/Shanghai",
        "EAT": "Africa/Addis_Ababa",
        "ECT": "Europe/Paris",
        "IET": "America/Indiana/Indianapolis",
        "IST": "Asia/Kolkata",
        "JST": "Asia/Tokyo",
        "MIT": "Pacific/Apia",
        "NET": "Asia/Yerevan",
        "NST": "Pacific/Auckland",
        "PLT": "Asia/Karachi",
        "PNT": "America/Phoenix",
        "PRT": "America/Puerto_Rico",
        "PST": "America/Los_Angeles",
        "SST": "Pacific/Guadalcanal",
        "VST": "Asia/Ho_Chi_Minh",
    }
    return timezone(TIMEZONE_ABBREVS.get(abbrev, abbrev))


def iter_except(func, exception, first=None):
    """ Call a function repeatedly until an exception is raised.

    Converts a call-until-exception interface to an iterator interface.
    Like builtins.iter(func, sentinel) but uses an exception instead
    of a sentinel to end the loop.
    """
    try:
        if first is not None:
            yield first()            # For database APIs needing an initial cast to db.first()
        while True:
            yield func()
    except exception:
        pass


def create_logger(name, level=logging.WARNING):
    """:Return: a logger with the given `name` and optional `level`."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


#############################################################


def main():
    """Simple test."""
    last_bid = 0.0
    last_ask = 0.0

    def finitize(x, replacement=0):
        return x if math.isfinite(x) else replacement

    def on_bar(instrument, bar):
        nonlocal last_bid, last_ask
        bar = Bar._make(map(finitize, bar))
        last_bid, last_ask = bar.bid, bar.ask
        timestamp = datetime.utcfromtimestamp(bar.time)
        if instrument.sec_type == 'FUT':
            print('{}\t{:.2f}/{:.2f}\t{:d}x{:d}\t{:d}@{:.2f}'.format(timestamp, bar.bid, bar.ask, int(bar.bidsize), int(bar.asksize), int(bar.volume), bar.vwap))
        elif instrument.sec_type == 'STK':
            print('{}\t{:.2f}/{:.2f}\t{:d}x{:d}\t{:d}@{:.2f}'.format(timestamp, bar.bid, bar.ask, int(bar.bidsize), int(bar.asksize), int(bar.volume) * 100, bar.vwap))
        elif instrument.sec_type == 'CASH':
            print('{}\t{:.5f}/{:.5f}\t{:d}x{:d}\t{:d}@{:.5f}'.format(timestamp, bar.bid, bar.ask, int(bar.bidsize), int(bar.asksize), int(bar.volume), bar.vwap))

    def on_order(order):
        print('order {} @ {}, profit ${:.2f}'.format(order.quantity, order.avg_price, order.profit))

    def on_alert(instrument, alert):
        print('ALERT {}: {}'.format(instrument, alert))

    ib = IBroke(verbose=4)
    #inst = ib.get_instrument("AAPL"); max_quantity = 200
    #inst = ib.get_instrument('EUR', 'CASH', 'IDEALPRO'); max_quantity = 20000
    inst = ib.get_instrument("ES", "FUT", "GLOBEX"); max_quantity = 1
    ib.register(inst, on_bar=on_bar, on_order=on_order, on_alert=on_alert, bar_size=5)
    try:
        for _ in range(20):
            time.sleep(5)
            if last_bid and last_ask:
                ib.cancel_all()
                time.sleep(3)
                ib.reconcile()
                quantity = max_quantity if random.random() > 0.5 else -max_quantity
                ib.order_target(inst, quantity, limit=last_bid if quantity > 0 else last_ask)
                print('pos {}, cost {}'.format(ib.get_position(inst), ib.get_cost(inst)))
    except KeyboardInterrupt:
        print('\nClosing...\n', file=sys.stderr)

    time.sleep(1)
    #ib.flatten()
    time.sleep(2)
    ib.disconnect()
    time.sleep(0.5)


class TestIBroke(unittest.TestCase):
    maxDiff = None

    def test_parse_trading_hours(self) -> None:
        vecs = (
            ('20170621:1700-1515,1530-1600;20170622:1700-1515,1530-1600', (
             (datetime(2017, 6, 21, 17, 00), datetime(2017, 6, 21, 15, 15)),
             (datetime(2017, 6, 21, 15, 30), datetime(2017, 6, 21, 16, 00)),
             (datetime(2017, 6, 22, 17, 00), datetime(2017, 6, 22, 15, 15)),
             (datetime(2017, 6, 22, 15, 30), datetime(2017, 6, 22, 16, 00)),
            )),
            ('20090507:0700-1830,1830-2330;20090508:CLOSED', (
             (datetime(2009, 5, 7, 7, 00), datetime(2009, 5, 7, 18, 30)),
             (datetime(2009, 5, 7, 18, 30), datetime(2009, 5, 7, 23, 30)),
            )),
            ('20170623:1715-1700;20170626:1715-1700', (
             (datetime(2017, 6, 23, 17, 15), datetime(2017, 6, 23, 17, 00)),
             (datetime(2017, 6, 26, 17, 15), datetime(2017, 6, 26, 17, 00)),
            )),
        )
        for timestr, dts in vecs:
            self.assertTupleEqual(tuple(Instrument._parse_trading_hours(timestr)), dts)

    def test_normalize_trading_hours(self) -> None:
        vecs = (
            ((
            (datetime(2017, 6, 21, 17, 00), datetime(2017, 6, 21, 15, 15)),
            (datetime(2017, 6, 21, 15, 30), datetime(2017, 6, 21, 16, 00)),
            (datetime(2017, 6, 22, 17, 00), datetime(2017, 6, 22, 15, 15)),
            (datetime(2017, 6, 22, 15, 30), datetime(2017, 6, 22, 16, 00)),
            ),
            (
            (datetime(2017, 6, 20, 17, 00, tzinfo=utc), datetime(2017, 6, 21, 15, 15, tzinfo=utc)),
            (datetime(2017, 6, 21, 15, 30, tzinfo=utc), datetime(2017, 6, 21, 16, 00, tzinfo=utc)),
            (datetime(2017, 6, 21, 17, 00, tzinfo=utc), datetime(2017, 6, 22, 15, 15, tzinfo=utc)),
            (datetime(2017, 6, 22, 15, 30, tzinfo=utc), datetime(2017, 6, 22, 16, 00, tzinfo=utc)),
            )),
        )
        for indates, outdates in vecs:
            self.assertTupleEqual(Instrument._normalize_trading_hours(indates, utc), outdates)


if __name__ == '__main__':
    main()
