from typing import Tuple

from pandas import DataFrame

from credentials import api, secret
from binance.client import Client
from dateutil.relativedelta import relativedelta
from datetime import datetime
import pandas as pd
import joblib as jb
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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


def gather_data(client: Client, pair: str = 'BTCUSDT', interval: str = None, period: str = None, start: str = None,
                stop: str = None) -> pd.DataFrame:
    """
    Gathers data

    :param client: Binance Client object
    :param pair: Pair for which data is extracted
    :param interval: Interval to consider
    :param period: Time period specified by a number followed by one of the options: 'y', 'm', 'd', 'h', 'min', 's'
    :param start: Date in a format of a string e.g. '2022-08-12'
    :param stop: Date in a format of a string e.g. '2023-08-13'
    :return: Dataframe containing financial data for chosen pair
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


def zero_replacement(data: pd.DataFrame, replacement: float = 1e-6) -> pd.DataFrame:
    """
    Replacing zeros with small values to avoid calculations with infinity

    :param data: Dataframe to be processes
    :param replacement: Value by which the zeros will be replaced
    :return: Dataframe with zeros replaced by 1e-6
    """
    tmp_ = data.copy()

    for column in tmp_.columns:
        tmp_[column] = tmp_[column].replace(0.0, replacement)

    return tmp_


def rates_of_return(data: pd.DataFrame) -> pd.DataFrame:
    """
    Rates of return preparation

    :param data: DataFrame with input values
    :return: Transformed dataframe with rates of return
    """

    tmp_ = pd.DataFrame()
    for column in data.columns:
        tmp_[column] = data[column].pct_change() * 100

    return tmp_.dropna()


def define_target(value: float) -> float:
    """
    Prepare the target variable

    :param value: Value on which the target variable is based
    :return: Target variable
    """

    # strong bull signal
    if value >= 1.0:
        result_ = 0
    # weak bull signal
    elif value >= 0.5:
        result_ = 1
    # consolidation phase
    elif value >= -0.5:
        result_ = 2
    # weak bear signal
    elif value >= -1.0:
        result_ = 3
    # strong bear signal
    else:
        result_ = 4

    return result_


def lag_series(series: pd.Series, lags: int = 24) -> pd.DataFrame:
    """
    Prepare lagged series

    :param series: Series representing target series
    :param lags: Integer representing number of lags to implement
    :return: Dataframe with target and lagged values
    """
    tmp_ = pd.DataFrame()

    for i in range(1, lags + 1):
        tmp_[f'{series.name}-lag-{i}'] = series.shift(i)

    tmp_.dropna(inplace=True)
    return tmp_.reset_index(drop=True)


def lag_dataframe(data: pd.DataFrame, target: str = 'close', lags: int = 24) -> pd.DataFrame:
    """
    Prepare lagged DataFrame

    :param data: Dataframe representing target DataFrame
    :param target: Series representing a target
    :param lags: Integer representing number of lags to implement
    :return: Dataframe with lagged values
    """
    tmp_list_ = []

    for column in data.columns:
        tmp_list_.append(lag_series(data[column], lags=lags))

    tmp_ = pd.concat(tmp_list_, axis=1)

    # target preparation
    tmp_target = data[target][lags:].reset_index(drop=True).copy()
    tmp_['target'] = tmp_target.apply(lambda x: define_target(x))

    return tmp_


def train_test_split(data: pd.DataFrame, split: float = 0.8) -> (pd.DataFrame, pd.DataFrame):
    """
    Split the data to train and test set

    :param data: Dataframe with target and lagged values
    :param split: Float value representing a ratio of the split
    :return: Tuple with data separated into two dataframes
    """
    tmp_ = data.copy()

    index_ = int(tmp_.shape[0] * split)

    train_ = tmp_[:index_].reset_index(drop=True)
    test_ = tmp_[index_:].reset_index(drop=True)

    return train_, test_


def normalize(x: pd.DataFrame, y: pd.DataFrame = None, exclude: list | None = None) -> (
        DataFrame | tuple[DataFrame, DataFrame]):
    """
    Normalize the dataframe(s)

    :param x: Dataframe to be transformed
    :param y: Dataframe to be transformed - optional
    :param exclude: Column names to exclude from parsing - will be added after transformation in the original form
    :return: Transformed dataframe(s)
    """

    scaler_ = MinMaxScaler()

    if y is None:
        if exclude is not None:
            tmp_x_ = x.drop(columns=exclude, axis=1)
        else:
            tmp_x_ = x.copy()

        scaler_.fit(tmp_x_)
        scaled_x_ = pd.DataFrame(scaler_.transform(tmp_x_), columns=tmp_x_.columns)

        for column in exclude:
            scaled_x_[column] = x[column]

        return scaled_x_
    else:
        if exclude is not None:
            tmp_x_ = x.drop(columns=exclude, axis=1)
            tmp_y_ = y.drop(columns=exclude, axis=1)
        else:
            tmp_x_ = x.copy()
            tmp_y_ = y.copy()

        scaler_.fit(tmp_x_)
        scaled_x_ = pd.DataFrame(scaler_.transform(tmp_x_), columns=tmp_x_.columns)
        scaled_y_ = pd.DataFrame(scaler_.transform(tmp_y_), columns=tmp_y_.columns)

        for column in exclude:
            scaled_x_[column] = x[column]
            scaled_y_[column] = y[column]

    # save the scaler_
    jb.dump(scaler_, 'scaler_.pkl')

    return scaled_x_, scaled_y_


def prepare_x_y(data: pd.DataFrame, target: str = 'target') -> (np.ndarray, np.ndarray):
    """
    Prepare X and y subsets

    :param data: Dataframe containing values
    :param target: Target column
    :return: Tuple containing desired format to input to the LSTM model
    """
    x_ = data.copy()
    y_ = x_.pop(target)

    x_values_ = x_.values
    y_values_ = y_.values

    reshaped_x_ = x_values_.reshape(x_values_.shape[0], 1, x_values_.shape[1])

    return reshaped_x_, y_values_
