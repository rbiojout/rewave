from __future__ import division,absolute_import,print_function
import numpy as np
import pandas as pd

import csv
import datetime
import h5py

from rewave.constants import eps

start_date = '2012-08-13'
end_date = '2017-08-11'
date_format = '%Y-%m-%d'
start_datetime = datetime.datetime.strptime(start_date, date_format)
end_datetime = datetime.datetime.strptime(end_date, date_format)
number_datetime = (end_datetime - start_datetime).days + 1


FEATURES_ADJUSTMENT = {"open":"adj_open",
                       "high":"adj_high",
                       "low":"adj_low",
                       "close":"adj_close",
                       "volume":"adj_volume"}

REVERTED_FEATURES_ADJUSTMENT = {"adj_open":"open",
                       "adj_high":"high",
                       "adj_low":"low",
                       "adj_close":"close",
                       "adj_volume":"volume"}


def translate_to(source, dict=FEATURES_ADJUSTMENT):
    """
    small utility class to translate from a dict, used for converting non adjusted to adjusted values
    :param dict: dictionary used
    :param source: array as source
    :return:
    """
    converted = []
    for element in source:
        converted.append(dict[element])
    return converted

def translate_from(converted, dict = FEATURES_ADJUSTMENT):
    """
    small utility class to reverse from a dict, used for converting adjusted to non adjusted values
    :param dict: dictionary used
    :param converted: array converted
    :return:
    """
    source = []
    for element in converted:
        source.append(list(dict.keys())[list(dict.values()).index(converted)])

def random_shift(x, fraction):
    """Apply a random shift to a pandas series."""
    min_x, max_x = np.min(x), np.max(x)
    m = np.random.uniform(-fraction, fraction, size=x.shape) + 1
    return np.clip(x * m, min_x, max_x)


def normalize(x):
    """Normalize to a pandas series."""
    x = (x - x.mean()) / (x.std() + eps)
    return x


def scale_to_start(x):
    """Scale pandas series so that it starts at one."""
    x = (x + eps) / (x[0] + eps)
    return x


def pricenorm3d(m, features, norm_method, fake_ratio=1.0, with_y=True):
    """normalize the price tensor, whose shape is [features, tickers, windowsize]
    @:param m: input tensor, unnormalized and there could be nan in it
    @:param with_y: if the tensor include y (future price)
        logging.debug("price are %s" % (self._latest_price_matrix[0, :, -1]))
    """
    result = m.copy()
    if features[0] != "close":
        raise ValueError("first feature must be close")
    for i, feature in enumerate(features):
        if with_y:
            one_position = 2
        else:
            one_position = 1
        pricenorm2d(result[i], m[0, :, -one_position], norm_method=norm_method,
                    fake_ratio=fake_ratio, one_position=one_position)
    return result


# input m is a 2d matrix, (tickernumber+1) * windowsize
def pricenorm2d(m, reference_column,
                norm_method="absolute", fake_ratio=1.0, one_position=2):
    if norm_method=="absolute":
        output = np.zeros(m.shape)
        for row_number, row in enumerate(m):
            if np.isnan(row[-one_position]) or np.isnan(reference_column[row_number]):
                row[-one_position] = 1.0
                for index in range(row.shape[0] - one_position + 1):
                    if index > 0:
                        row[-one_position - index] = row[-index - one_position + 1] / fake_ratio
                row[-one_position] = 1.0
                row[-1] = fake_ratio
            else:
                row = row / reference_column[row_number]
                for index in range(row.shape[0] - one_position + 1):
                    if index > 0 and np.isnan(row[-one_position - index]):
                        row[-one_position - index] = row[-index - one_position + 1] / fake_ratio
                if np.isnan(row[-1]):
                    row[-1] = fake_ratio
            output[row_number] = row
        m[:] = output[:]
    elif norm_method=="relative":
        output = m[:, 1:]
        divisor = m[:, :-1]
        output = output / divisor
        pad = np.empty((m.shape[0], 1,))
        pad.fill(np.nan)
        m[:] = np.concatenate((pad, output), axis=1)
        m[np.isnan(m)] = fake_ratio
    else:
        raise ValueError("there is no norm morthod called %s" % norm_method)

def prepare_dataframe(df):
    """
    Modify the dataframe from two dimension to a dataframe for the network
    :param df: dataframe in 2D, with the MultiIndex in the colums (level 0: assets, level 1: features)
    :return: np array in 3D : assets, time, features
    """
    asset_names = df.columns.levels[0].tolist()
    features = df.columns.levels[1].tolist()
    data = df.as_matrix().reshape(
        (len(df), len(asset_names), len(features)))
    # order assets, time, features
    return np.transpose(data, (1, 0, 2))

def _pack_samples(self, tensor3D):
    """
    take a 3D tensor compose of assets, window_time, features
    and return a 3D tensor with the differences
    return the elements for the indexs
    :param indexs: indexs from which we extract
    :return: inputs, outputs, control functions
    """
    tensor3D[:,:,:]/tensor3D[:,:,:].shift(1)

    indexs = np.array(indexs)

    last_w = self.PVM.values[indexs-1, :]

    def setw(w):
        self.PVM.iloc[indexs, :] = w
    M = [self.get_submatrix(index) for index in indexs]
    M = np.array(M)
    X = M[:, :, :-1, :]
    y = M[:, :, -1, :] / M[:, 0, None, -2, :]

    return {"X": X, "y": y, "last_w": last_w, "setw": setw}

def get_chart_until_success(polo, pair, start, period, end):
    is_connect_success = False
    chart = {}
    while not is_connect_success:
        try:
            chart = polo.marketChart(pair=pair, start=int(start), period=int(period), end=int(end))
            is_connect_success = True
        except Exception as e:
            print(e)
    return chart


def get_type_list(feature_number):
    """
    :param feature_number: an int indicates the number of features
    :return: a list of features n
    """
    if feature_number == 1:
        type_list = ["close"]
    elif feature_number == 2:
        type_list = ["close", "volume"]
        raise NotImplementedError("the feature volume is not supported currently")
    elif feature_number == 3:
        type_list = ["close", "high", "low"]
    elif feature_number == 4:
        type_list = ["close", "high", "low", "open"]
    else:
        raise ValueError("feature number could not be %s" % feature_number)
    return type_list

def get_ticker_list(asset_number):
    """
    :param asset_number: an int indicates the number of list
    :return: a list of features n
    """
    if asset_number == 1:
        ticker_list = ("AAPL", "MSFT", "ABC")
    elif asset_number == 2:
        ticker_list = ("AAPL", "MSFT", "ABC", "ABT", "AIG", "AEE", "ABBV", "AEP", "ADSK", "AES")
        # raise NotImplementedError("the feature volume is not supported currently")
    else:
        raise ValueError("feature number could not be %s" % asset_number)
    return ticker_list


def panel2array(panel):
    """convert the panel to datatensor (numpy array) without btc
    """
    without_btc = np.transpose(panel.values, axes=(2, 0, 1))
    return without_btc


def count_periods(start, end, period_length):
    """
    :param start: unix time, excluded
    :param end: unix time, included
    :param period_length: length of the period
    :return: 
    """
    return (int(end)-int(start)) // period_length

def get_volume_forward(time_span, portion, portion_reversed):
    volume_forward = 0
    if not portion_reversed:
        volume_forward = time_span*portion
    return volume_forward


def panel_fillna(panel, type="bfill"):
    """
    fill nan along the 3rd axis
    :param panel: the panel to be filled
    :param type: bfill or ffill
    """
    frames = {}
    for item in panel.items:
        if type == "both":
            frames[item] = panel.loc[item].fillna(axis=1, method="bfill").\
                fillna(axis=1, method="ffill")
        else:
            frames[item] = panel.loc[item].fillna(axis=1, method=type)
    return pd.Panel(frames)

def dataframe_fillna(df, type="bfill"):
    """
    fill nan along the 2nd axis
    :param df: the dataframe to be filled
    :param type: bfill or ffill
    """
    if type == "both":
        df = df.fillna(axis=1, method="bfill"). \
            fillna(axis=1, method="ffill")
    else:
        df = df.fillna(axis=1, method=type)
    return df

def normalize(x):
    """ Create a universal normalization function across close/open ratio

    Args:
        x: input of any shape

    Returns: normalized data

    """
    return (x - 1) * 100


def index_to_date(index):
    """

    Args:
        index: the date from start-date (2012-08-13)

    Returns:

    """
    return (start_datetime + datetime.timedelta(index)).strftime(date_format)


def date_to_index(date_string):
    """

    Args:
        date_string: in format of '2012-08-13'

    Returns: the days from start_date: '2012-08-13'

    >>> date_to_index('2012-08-13')
    0
    >>> date_to_index('2012-08-12')
    -1
    >>> date_to_index('2012-08-15')
    2
    """
    return (datetime.datetime.strptime(date_string, date_format) - start_datetime).days

def create_optimal_imitation_dataset2(history, training_data_ratio=0.8):
    """

    :param history: a list of frames with dimension nb_assets, window_length, features
    :param training_data_ratio: split ration
    :return: dataset for further work
    """
    nb_samples = len(history)
    Xs = []
    Ys = []
    for i in range(nb_samples):
        frame = history[i]
        obs = frame[:, :-2, :]
        label = np.zeros(dtype=np.float32, shape=(frame.shape[0],))
        max_index = np.argmax(frame[:, -2, 0], axis=0)
        label[max_index] = 1.0
        Xs.append(obs)
        Ys.append(label)
    Xs = np.stack(Xs, axis=0)
    Ys = np.stack(Ys, axis=0)
    num_training_sample = int(nb_samples * training_data_ratio)
    return (Xs[:num_training_sample], Ys[:num_training_sample]), \
           (Xs[num_training_sample:], Ys[num_training_sample:])

def create_optimal_imitation_dataset(X_full, y_full, training_data_ratio=0.8):
    """

    :param history: a list of frames with dimension nb_assets, window_length, features
    :param training_data_ratio: split ration
    :return: dataset for further work
    """
    nb_samples = len(X_full)
    Xs = []
    Ys = []
    for i in range(nb_samples):
        obs = X_full[i]
        label = np.zeros(dtype=np.float32, shape=(obs.shape[0],))
        max_index = np.argmax(y_full[i], axis=0)
        label[max_index] = 1.0
        Xs.append(obs)
        Ys.append(label)
    Xs = np.stack(Xs, axis=0)
    Ys = np.stack(Ys, axis=0)
    num_training_sample = int(nb_samples * training_data_ratio)
    return (Xs[:num_training_sample], Ys[:num_training_sample]), \
           (Xs[num_training_sample:], Ys[num_training_sample:])


def create_optimal_imitation_dataset_old(history, training_data_ratio=0.8, is_normalize=True):
    """ Create dataset for imitation optimal action given future observations

    Args:
        history: size of (num_stocks, T, num_features) contains (open, high, low, close)
        training_data_ratio: the ratio of training data

    Returns: un-normalized close/open ratio with size (T, num_stocks), labels: (T,)
             split the data according to training_data_ratio

    """
    num_stocks, T, num_features = history.shape
    cash_history = np.ones((1, T, num_features))
    history = np.concatenate((cash_history, history), axis=0)
    close_open_ratio = np.transpose(history[:, :, 3] / history[:, :, 0])
    if is_normalize:
        close_open_ratio = normalize(close_open_ratio)
    labels = np.argmax(close_open_ratio, axis=1)
    num_training_sample = int(T * training_data_ratio)
    return (close_open_ratio[:num_training_sample], labels[:num_training_sample]), \
           (close_open_ratio[num_training_sample:], labels[num_training_sample:])

def create_imitation_dataset(history, training_data_ratio=0.8):
    """

    :param history: a list of frames with dimension nb_assets, window_length, features
    :param training_data_ratio: split ration
    :return: dataset for further work
    """
    nb_samples = len(history)
    Xs = []
    Ys = []
    for i in range(nb_samples):
        frame = history[i]
        obs = frame[:, :-1, :]
        label = np.argmax(frame[:, -1, 0], axis=0)
        Xs.append(obs)
        Ys.append(label)
    Xs = np.stack(Xs)
    Ys = np.concatenate(Ys)
    num_training_sample = int(nb_samples * training_data_ratio)
    return (Xs[:num_training_sample], Ys[:num_training_sample]), \
           (Xs[num_training_sample:], Ys[num_training_sample:])

def create_imitation_dataset_old(history, window_length, training_data_ratio=0.8, is_normalize=True):
    """ Create dataset for imitation optimal action given past observations

    Args:
        history: size of (num_stocks, T, num_features) contains (open, high, low, close)
        window_length: length of window as feature
        training_data_ratio: for splitting training data and validation data
        is_normalize: whether to normalize the data

    Returns: close/open ratio of size (num_samples, num_stocks, window_length)

    """
    num_stocks, T, num_features = history.shape
    cash_history = np.ones((1, T, num_features))
    history = np.concatenate((cash_history, history), axis=0)
    close_open_ratio = history[:, :, 3] / history[:, :, 0]
    if is_normalize:
        close_open_ratio = normalize(close_open_ratio)
    Xs = []
    Ys = []
    for i in range(window_length, T):
        obs = close_open_ratio[:, i - window_length:i]
        label = np.argmax(close_open_ratio[:, i:i+1], axis=0)
        Xs.append(obs)
        Ys.append(label)
    Xs = np.stack(Xs)
    Ys = np.concatenate(Ys)
    num_training_sample = int(T * training_data_ratio)
    return (Xs[:num_training_sample], Ys[:num_training_sample]), \
           (Xs[num_training_sample:], Ys[num_training_sample:])
