from credentials import api, secret
from binance.client import Client
from dateutil.relativedelta import relativedelta
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def market_connection(api_key: str = api, secret_key: str = secret) -> Client:
    """
    Makes a connection with binance API

    :param api_key: Binance API key
    :param secret_key: Binance API secret key
    :return: Binance Client object
    """

    return Client(api_key, secret_key)


def data_range(period: str = None, start: str = None, stop: str = None) -> (int, int):
    """
    Sets a correct data range

    :param period: Time period specified by a number followed by one of the options: 'y', 'm', 'd', 'h', 'min', 's'
    :param start: Date in a format of a string e.g. '2022-08-12'
    :param stop: Date in a format of a string e.g. '2023-08-13'
    :return: start, stop in milliseconds
    """

    try:
        if period is not None:
            if start is None and stop is None:
                count_ = int(period[:-1])
                time_ = period[-1]
                stop_ = pd.to_datetime(datetime.today().date())

                if time_ == 'y':
                    start_ = stop_ - relativedelta(years=count_)
                elif time_ == 'm':
                    start_ = stop_ - relativedelta(months=count_)
                elif time_ == 'd':
                    start_ = stop_ - relativedelta(days=count_)
                elif time_ == 'h':
                    start_ = stop_ - relativedelta(hours=count_)
                elif time_ == 'min':
                    start_ = stop_ - relativedelta(minutes=count_)
                elif time_ == 's':
                    start_ = stop_ - relativedelta(seconds=count_)
                else:
                    raise KeyError("Invalid period")
            else:
                raise KeyError("Period can't be set together with start or stop")

        else:
            # TODO add period for certain points in time
            if start is None and stop is None:
                return data_range(period='1y')
            elif start is not None and stop is not None:
                start_ = start
                stop_ = stop
            else:
                raise KeyError("Specifying both start and stop is necessary")

        return int(start_.value / 1e6), int(stop_.value / 1e6)
    except Exception as e:
        raise Exception(e)


def data_gathering(client: Client, pair: str = 'BTCUSDT', interval: str = None, period: str = None, start: str = None,
                   stop: str = None) -> pd.DataFrame:
    """
    Gathers data

    :param client: Binance Client object
    :param pair: Pair for which data is extracted
    :param interval: Interval to consider
    :param period: Time period specified by a number followed by one of the options: 'y', 'm', 'd', 'h', 'min', 's'
    :param start: Date in a format of a string e.g. '2022-08-12'
    :param stop: Date in a format of a string e.g. '2023-08-13'
    :return: pd.DataFrame containing financial data for chosen pair
    """

    if interval is None:
        interval = client.KLINE_INTERVAL_1HOUR

    start_, stop_ = data_range(period=period, start=start, stop=stop)

    list_of_trades_ = []
    while True:
        trades_ = client.get_klines(symbol=pair, interval=interval, startTime=start_, endTime=stop_, limit=1000)
        list_of_trades_.extend(trades_[:-1])
        if trades_[-1][0] == stop_:
            break
        else:
            start_ = trades_[-1][0]

    frame_ = pd.DataFrame(list_of_trades_,
                          columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset',
                                   'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                   'ignore'])

    filtered_frame_ = pd.DataFrame()

    filtered_frame_['open_time'] = pd.to_datetime(frame_['open_time'], unit='ms')
    filtered_frame_['close_time'] = pd.to_datetime(frame_['close_time'], unit='ms')

    for name_ in ['open', 'high', 'low', 'close', 'volume']:
        filtered_frame_[name_] = pd.to_numeric(frame_[name_])

    return filtered_frame_


# def indicators_creation(data: pd.DataFrame, ) -> pd.DataFrame:
#     """
#     Creates variables
#
#     :param data: pd.DataFrame containing financial data for chosen pair
#     :return: pd.DataFrame with indicators
#     """
#
#     open_ = data['open']
#     high_ = data['high']
#     low_ = data['low']
#     close_ = data['close']
#     volume_ = data['volume']
#
#     indicators_frame_ = pd.DataFrame()
#
#     return indicators_frame_


# def features_creation(data: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
#     """
#     Create meaningful features
#
#     :param data: pd.DataFrame containing financial data for chosen pair
#     :param indicators: pd.DataFrame with indicators
#     :return: pd.DataFrame with new features
#     """
#
#     features_frame_ = pd.DataFrame()
#
#     return features_frame_


def split_data(data: pd.DataFrame, split: float = 0.7) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits data two train and test set

    :param data: pd.DataFrame containing financial data for chosen pair
    :param split: Float value of a split from range (0, 1)
    :return: Two pd.DataFrames
    """
    index_ = int(data.shape[0] * split)

    train_ = data.iloc[:index_, 2:]
    test_ = data.iloc[index_:, 2:]

    train_ = train_.reset_index().drop(columns=['index'])
    test_ = test_.reset_index().drop(columns=['index'])

    return train_, test_


def input_preparation(data: pd.DataFrame, lag: int = 60) -> (np.array, np.array, MinMaxScaler):
    """
    Prepares the data input

    :param data: pd.DataFrame containing financial data for chosen pair
    :param lag: Number of periods to delay
    :return: Transformed x and y arrays and a scaler
    """
    target_index_ = None
    for index, name in enumerate(data.columns):
        if name == 'close':
            target_index_ = index
            break

    if target_index_ is None:
        raise KeyError("No target column")

    scaler_ = MinMaxScaler(feature_range=(0, 1))
    scaled_set_ = scaler_.fit_transform(data)
    x_ = []
    y_ = []
    for i in range(lag, data.shape[0]):
        x_.append(scaled_set_[i - lag:i])
        y_.append(scaled_set_[i, target_index_])

    x_, y_ = np.array(x_), np.array(y_)

    return x_, y_, scaler_
