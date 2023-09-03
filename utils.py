from credentials import api, secret
from binance.client import Client


def market_connection(api_key=api, secret_key=secret):
    return Client(api_key, secret_key)


def data_gathering(client, pair='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR, period=None, start=None, end=None):
    # depth = client.get_order_book(symbol=pair)  # market depth

    if period is not None:
        trades = client.get_klines(symbol=pair, interval=interval)


    return trades


def run():
    client = market_connection()

    data = data_gathering(client)

    return data
