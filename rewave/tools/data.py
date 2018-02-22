from __future__ import division,absolute_import,print_function
import numpy as np
import pandas as pd

from rewave.constants import eps


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

