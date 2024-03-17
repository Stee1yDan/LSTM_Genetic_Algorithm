import random

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_percentage_error, \
    root_mean_squared_error, max_error


class ModelWrapper:
    def __init__(self, has_second_layer, has_third_layer, hyperparameters, layer_1_neurons, layer_2_neurons,
                 layer_3_neurons, dense_number, dropout_rate, optimizer, prediction_units, future_units, loss):
        self.scaler = None
        self.score = None
        self.y_test = None
        self.x_test = None
        self.y_train = None
        self.x_train = None
        self.model = None
        self.has_second_layer = has_second_layer
        self.has_third_layer = has_third_layer
        self.hyperparameters = hyperparameters
        self.layer_1_neurons = layer_1_neurons
        self.layer_2_neurons = layer_2_neurons
        self.layer_3_neurons = layer_3_neurons
        self.dense_number = dense_number
        self.optimizer = optimizer
        self.prediction_units = prediction_units
        self.future_units = future_units
        self.dropout_rate = dropout_rate
        self.loss = loss

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.prediction_units, len(self.hyperparameters))))
        model.add(LSTM(units=self.layer_1_neurons, return_sequences=True))
        model.add(Dropout(self.dropout_rate))

        if self.has_second_layer:
            model.add(LSTM(units=self.layer_2_neurons, return_sequences=True))
            model.add(Dropout(self.dropout_rate))

        if self.has_third_layer:
            model.add(LSTM(units=self.layer_3_neurons))
            model.add(Dropout(self.dropout_rate))

        model.add(Dense(units=self.dense_number))

        model.add(Dense(units=1))

        model.compile(optimizer=self.optimizer, loss=self.loss)
        model.fit(self.x_train, self.y_train, epochs=10, batch_size=128)

        self.model = model

    def prepare_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(
            data[self.hyperparameters].values.reshape(-1, len(self.hyperparameters)))
        res_data = self.scaler.fit_transform(data[['Close']].values.reshape(-1, 1))

        prediction_units = self.prediction_units
        future_units = self.future_units

        x_full = []
        y_full = []

        for x in range(prediction_units, len(scaled_data) - future_units):
            x_full.append(scaled_data[x - prediction_units:x])
            y_full.append(res_data[x + future_units, 0])

        x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2)

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_test, y_test = np.array(x_test), np.array(y_test)

        self.x_train = x_train.reshape(-1, x_train.shape[1], len(self.hyperparameters))
        self.y_train = y_train

        self.x_test = x_test.reshape(-1, x_test.shape[1], len(self.hyperparameters))
        self.y_test = y_test

    def predict(self):
        predictions = self.model.predict(self.x_test)
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))

        temp = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        self.score = r2_score(temp, predictions)

    def to_string(self):
        print("ModelWrapper instance with:")
        print("\t-- Has second layer:", self.has_second_layer)
        print("\t-- Has third layer:", self.has_third_layer)
        print("\t-- Hyperparameters:", self.hyperparameters)
        print("\t-- Neurons in layer 1:", self.layer_1_neurons)
        print("\t-- Neurons in layer 2:", self.layer_2_neurons)
        print("\t-- Neurons in layer 3:", self.layer_3_neurons)
        print("\t-- Dense neurons number:", self.dense_number)
        print("\t-- Optimizer:", self.optimizer)
        print("\t-- Prediction units:", self.prediction_units)
        print("\t-- Future units:", self.future_units)
        print("\t-- Dropout rate:", self.dropout_rate)
        print("\t-- Loss:", self.loss)
        print("\t-- Score:", self.score)


def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2score = r2_score(y_test, y_pred)
    msle = mean_squared_log_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    max_err = max_error(y_test, y_pred)

    print("MSE :", mse)
    print("RMSE: ", rmse)
    print("MSLE: ", msle)
    print("MAPE: ", mape)
    print("Max_error: ", max_err)
    print("R2_score: ", r2score)


def make_child(parent_1: ModelWrapper, parent_2: ModelWrapper):
    parents_hyperparameters = list(set(parent_1.hyperparameters + parent_2.hyperparameters))
    current_hyperparameters = list(set(["Close", "Open"] + random.sample(parents_hyperparameters,
                                                                         random.randint(2,
                                                                                        len(parents_hyperparameters)))))

    return ModelWrapper(random.choice([parent_1.has_second_layer, parent_2.has_second_layer]),
                        random.choice([parent_1.has_third_layer, parent_2.has_third_layer]),
                        current_hyperparameters,
                        random.choice([parent_1.layer_1_neurons, parent_2.layer_1_neurons]),
                        random.choice([parent_1.layer_1_neurons, parent_2.layer_1_neurons]),
                        random.choice([parent_1.layer_1_neurons, parent_2.layer_1_neurons]),
                        int((parent_1.dense_number + parent_2.dense_number) / 2),
                        random.choice([parent_1.dropout_rate, parent_2.dropout_rate]),
                        random.choice([parent_1.optimizer, parent_2.optimizer]),
                        7,
                        1,
                        random.choice([parent_1.loss, parent_2.loss]))


data = yf.download('AAPL', '2020-01-12', '2024-01-12', interval='1d')

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

optimizers = ["adagrad", "adam", "adamax", "rmsprop", "sgd"]
possible_hyperparameters = ["RSI", "ATR", "Signal_Line", "pct_change", "log_returns"]
loss_functions = ["mse", "mae"]
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
neurons_in_layer = [8, 16, 32, 64, 128, 256]


def start_genetic_algorithm():
    generation = []

    for i in range(20):
        current_hyperparameters = ["Close", "Open"] + random.sample(possible_hyperparameters,
                                                                    random.randint(1, len(possible_hyperparameters)))

        model = ModelWrapper(random.choice([True, False]),
                             random.choice([True]),
                             current_hyperparameters,
                             random.choice(neurons_in_layer),
                             random.choice(neurons_in_layer),
                             random.choice(neurons_in_layer),
                             random.randint(1, 24),
                             random.choice(dropout_rates),
                             random.choice(optimizers),
                             7,
                             1,
                             random.choice(loss_functions))

        model.prepare_data()
        model.build_model()
        model.predict()
        model.to_string()

        generation.append(model)

    generation.sort(key=lambda x: x.score, reverse=True)

    for gen in range(5):
        generation = generation[0:5]

        print("Current generation: " + str(gen))
        print("Current winner is: " + str(generation[0].score))
        new_generation = []

        for i in range(0, len(generation)):
            for j in range(i, len(generation)):
                new_model = make_child(generation[i], generation[j])
                new_model.prepare_data()
                new_model.build_model()
                new_model.predict()
                new_model.to_string()


                new_generation.append(new_model)

        new_generation.sort(key=lambda x: x.score, reverse=True)
        generation = new_generation


    print("-----Winner-----")
    overall_winner = generation[0]
    overall_winner.to_string()


start_genetic_algorithm()
