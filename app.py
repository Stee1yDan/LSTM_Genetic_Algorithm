from model_prediction_handler import get_prediction, write_data_to_sheet
from modeldev import start_genetic_algorithm, ModelParameters, ModelWrapper
from flask import Flask, request, json
import __main__

__main__.ModelParameters = ModelParameters

app = Flask(__name__)
your_rest_server_port = 5000


@app.get("/api/v1/models/predict/<ticker>")
def get_full_stock_info(ticker: str):
    prediction = get_prediction(ticker)
    return str(prediction[-1][0])
