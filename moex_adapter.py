import requests

import apimoex
import pandas as pd

ru_tickers = ["VTBR", "LKOH", "ASTR", "GMKN", "MGNT", "PLZL", "GAZP", "SBER", "NVTK", "YNDX"]

request_url = ('https://iss.moex.com/iss/engines/stock/'
               'markets/shares/boards/TQBR/securities.json')
arguments = {'securities.columns': ('SECID,'
                                    'REGNUMBER,'
                                    'LOTSIZE,'
                                    'SHORTNAME')}

def get_stock_info(ticker):
    obj = {}
    with requests.Session() as session:
        df = apimoex.find_security_description(session, ticker)
        values = ["NAME", "SHORTNAME"]
        for d in df:
            if d["name"] in values:
                obj[d["name"]] = d["value"]
    return obj



def get_historical_data(ticker, start, end, interval):

    if interval == "1mo": interval = 31
    if interval == "1w": interval = 7
    if interval == "1d": interval = 24
    if interval == "1h": interval = 1

    with requests.Session() as session:
        data = apimoex.get_market_candles(session, ticker, interval=interval, start=start, end=end)

        columns = ['Date', 'Close', 'Open', 'Low', 'High', 'Volume']

        df = pd.DataFrame(columns=columns)

        for d in data:
            df.loc[-1] = [d['begin'], d['close'], d['open'], d['low'], d['high'], d['volume']]
            df.index = df.index + 1
            df = df.sort_index()

        df = df.set_index('Date')

        return df
