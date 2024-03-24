import json
import numpy as np
import pandas as pd
import xlsxwriter as xlsx
import tensorflow as tf
from tensorflow import keras
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler


class ModelParameters:
    def __init__(self, ticker, has_second_layer, hyperparameters, layer_1_neurons, layer_2_neurons,
                 layer_3_neurons, dense_number, dropout_rate, optimizer, loss, score):
        self.ticker = ticker
        self.has_second_layer = has_second_layer
        self.hyperparameters = hyperparameters
        self.layer_1_neurons = layer_1_neurons
        self.layer_2_neurons = layer_2_neurons
        self.layer_3_neurons = layer_3_neurons
        self.dense_number = dense_number
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss = loss
        self.score = score

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True)


def deserialize_modelparameters(deserialized_data):
    ticker = deserialized_data['ticker']
    has_second_layer = deserialized_data['has_second_layer']
    hyperparameters = deserialized_data['hyperparameters']
    layer_1_neurons = deserialized_data['layer_1_neurons']
    layer_2_neurons = deserialized_data['layer_2_neurons']
    layer_3_neurons = deserialized_data['layer_3_neurons']
    dense_number = deserialized_data['dense_number']

    dropout_rate = deserialized_data['dropout_rate']
    optimizer = deserialized_data['optimizer']
    loss = deserialized_data['loss']
    score = deserialized_data['score']

    return ModelParameters(ticker, has_second_layer, hyperparameters, layer_1_neurons, layer_2_neurons,
                         layer_3_neurons, dense_number, dropout_rate, optimizer, loss, score)


workbook = xlsx.Workbook('models.xlsx')
worksheet = workbook.add_worksheet()

tickers = ["AAPL", "MSFT", "TSLA", "CAT", "KO", "MCD", "SBUX", "AMZN", "ADBE", "CMCSA", "JNJ", "DIS", "INTC",
                      "NVDA",
                      "AMD", "PFE", "PEP", "GM", "BA", "RACE", "META", "GOOG", "GE", "GD", "MA", "ORCL", "PYPL",
                      "NKE",
                      "AXP", "BEN", "BKNG", "CL", "DAL", "DPZ", "EA", "V", "WAB", "XEL", "YUM"]

row_param = ["layer_1", "layer_2", "layer_3", "dense_layer", "dropout_rate", "loss", "optimizer", "RSI",
                        "ATR",
                        "Signal_Line",
                        "pct_change", "log_returns", "score"]


def prepare_workbook():
    global tickers
    global row_param
    global workbook
    global worksheet

    worksheet.write_column(1, 0, tickers)
    worksheet.write_row(0, 1, row_param)


def write_data_to_sheet(model_parameters: ModelParameters):
    global tickers
    global row_param
    global workbook
    global worksheet

    model_row = tickers.index(model_parameters.ticker) + 1

    worksheet.write(model_row, row_param.index("layer_1") + 1, model_parameters.layer_1_neurons)

    if model_parameters.has_second_layer:
        worksheet.write(model_row, row_param.index("layer_2") + 1, model_parameters.layer_2_neurons)
    else:
        worksheet.write(model_row, row_param.index("layer_2") + 1, 0)

    worksheet.write(model_row, row_param.index("layer_3") + 1, model_parameters.layer_3_neurons)
    worksheet.write(model_row, row_param.index("dense_layer") + 1, model_parameters.dense_number)
    worksheet.write(model_row, row_param.index("dropout_rate") + 1, model_parameters.dropout_rate)
    worksheet.write(model_row, row_param.index("loss") + 1, model_parameters.loss)
    worksheet.write(model_row, row_param.index("optimizer") + 1, model_parameters.optimizer)
    worksheet.write(model_row, row_param.index("score") + 1, model_parameters.score)

    worksheet.write(model_row, row_param.index("RSI") + 1, "RSI" in model_parameters.hyperparameters)
    worksheet.write(model_row, row_param.index("ATR") + 1, "ATR" in model_parameters.hyperparameters)
    worksheet.write(model_row, row_param.index("Signal_Line") + 1, "Signal_Line" in model_parameters.hyperparameters)
    worksheet.write(model_row, row_param.index("pct_change") + 1, "pct_change" in model_parameters.hyperparameters)
    worksheet.write(model_row, row_param.index("log_returns") + 1, "log_returns" in model_parameters.hyperparameters)


def download_data(ticker: str):
    return yf.download(ticker, start=(datetime.today() - relativedelta(days=53)).strftime('%Y-%m-%d'),
                       end=datetime.today().strftime('%Y-%m-%d'), interval="1d")


def prepare_data(data):
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data["HL"] = data["High"] - data["Low"]
    data["HC"] = abs(data["High"] - data["Close"].shift())
    data["LC"] = abs(data["Low"] - data["Close"].shift())
    data["TR"] = data[["HL", "HC", "LC"]].max(axis=1)
    data["ATR"] = data["TR"].rolling(window=14).mean()

    delta = data["Close"].diff(1)
    delta = delta.dropna()
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(window=14).mean()
    average_loss = abs(down.rolling(window=14).mean())
    rs = average_gain / average_loss
    RSI = 100.0 - (100.0 / (1.0 + rs))

    data["RSI"] = RSI

    data = data.iloc[14:]

    data['pct_change'] = data.Close.pct_change()
    data['log_returns'] = np.log(1 + data['pct_change'])

    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data["HL"] = data["High"] - data["Low"]
    data["HC"] = abs(data["High"] - data["Close"].shift())
    data["LC"] = abs(data["Low"] - data["Close"].shift())
    data["TR"] = data[["HL", "HC", "LC"]].max(axis=1)
    data["ATR"] = data["TR"].rolling(window=14).mean()

    delta = data["Close"].diff(1)
    delta = delta.dropna()
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(window=14).mean()
    average_loss = abs(down.rolling(window=14).mean())
    rs = average_gain / average_loss
    RSI = 100.0 - (100.0 / (1.0 + rs))

    data["RSI"] = RSI

    data = data.iloc[14:]

    return data


def fill_the_workbook():
    global tickers
    prepare_workbook()
    for ticker in tickers:
        model_parameters: ModelParameters
        with open(ticker + ".json", "r") as f:
            temp = json.loads(json.load(f))
            model_parameters = deserialize_modelparameters(temp)
        write_data_to_sheet(model_parameters)
        print(ticker + ": " + str(model_parameters.score))
    workbook.close()


def get_prediction(ticker: str):
    data = prepare_data(download_data(ticker))
    model = tf.keras.models.load_model("./models/" + ticker + "_model.keras")
    model_parameters: ModelParameters
    with open("./models/" + ticker + ".json", "r") as f:
        temp = json.loads(json.load(f))
        model_parameters = deserialize_modelparameters(temp)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(
        data[model_parameters.hyperparameters].values.reshape(-1, len(model_parameters.hyperparameters)))
    res_data = scaler.fit_transform(data[['Close']].values.reshape(-1, 1))

    x_full = []

    prediction_units = 7
    future_units = 1

    for x in range(prediction_units, len(scaled_data) - future_units):
        x_full.append(scaled_data[x - prediction_units:x])

    x_full = np.array(x_full)

    x_full = x_full.reshape(-1, x_full.shape[1], len(model_parameters.hyperparameters))

    res = model.predict(x_full)
    res = scaler.inverse_transform(res.reshape(-1, 1))

    return res[-1][0]