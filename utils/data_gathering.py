from binance.client import Client
from dateutil.relativedelta import relativedelta
from datetime import datetime
import pandas as pd
import os

API = os.environ.get("API")
SECRET = os.environ.get("SECRET")


def market_connection(api_key: str = API, secret_key: str = SECRET) -> Client:
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
            if start is None and stop is None:
                return data_range(period='1y')
            elif start is not None and stop is not None:
                start_ = pd.to_datetime(start)
                stop_ = pd.to_datetime(stop)
            else:
                raise KeyError("Specifying both start and stop is necessary")

        return int(start_.value / 1e6), int(stop_.value / 1e6)
    except Exception as e:
        raise Exception(e)


def gather_data(client: Client = market_connection(), pair: str = 'BTCUSDT', interval: str = None, period: str = None,
                start: str = None,
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
        interval = Client.KLINE_INTERVAL_1HOUR

    start_, stop_ = data_range(period=period, start=start, stop=stop)
    list_of_trades_ = []

    # TODO psuje się dla interval=Client.KLINE_INTERVAL_1DAY, start='2023-01-01 00:00:00', stop='2023-12-31 23:00:00'
    while True:
        trades_ = client.get_klines(symbol=pair, interval=interval, startTime=start_, endTime=stop_, limit=1000)
        list_of_trades_.extend(trades_[:-1])
        if trades_[-1][0] == stop_:
            break
        else:
            start_ = trades_[-1][0]

    frame_ = pd.DataFrame(list_of_trades_,
                          columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset',
                                   'n_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                   'ignore'])

    filtered_frame_ = pd.DataFrame()

    filtered_frame_['open_time'] = pd.to_datetime(frame_['open_time'], unit='ms')
    filtered_frame_['close_time'] = pd.to_datetime(frame_['close_time'], unit='ms')

    for name_ in ['open', 'high', 'low', 'close', 'volume', 'n_trades']:
        filtered_frame_[name_] = pd.to_numeric(frame_[name_])

    return filtered_frame_
