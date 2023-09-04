from credentials import api, secret
from binance.client import Client
from dateutil.relativedelta import relativedelta
from datetime import datetime
import pandas as pd


def market_connection(api_key=api, secret_key=secret):
    return Client(api_key, secret_key)


def data_range(period=None, start=None, stop=None):
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


def data_gathering(client, pair='BTCUSDT', interval=None, period=None, start=None, stop=None):
    if interval is None:
        interval = client.KLINE_INTERVAL_1HOUR

    start, stop = data_range(period=period, start=start, stop=stop)

    trades = client.get_klines(symbol=pair, interval=interval, startTime=start, endTime=stop, limit=1000)




    list_of_trades = []
    while True:
        trades = client.get_klines(symbol=pair, interval=interval, startTime=start, endTime=stop, limit=1000)
        list_of_trades.extend(trades[:-1])
        if trades[-1][0] == stop:
            break
        else:
            start = trades[-1][0]

    frame = pd.DataFrame(list_of_trades,
                         columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset',
                                  'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                  'ignore'])
    filtered_frame = frame[['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume']]
    filtered_frame['open_time'] = pd.to_datetime(filtered_frame['open_time'], unit='ms')
    filtered_frame['close_time'] = pd.to_datetime(filtered_frame['close_time'], unit='ms')

    return filtered_frame


def run():
    client = market_connection()

    data = data_gathering(client,interval=client.KLINE_INTERVAL_1HOUR, period='1d')

    return data
