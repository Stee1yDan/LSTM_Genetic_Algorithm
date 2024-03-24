from model_prediction_handler import get_prediction, ModelParameters
from flask import Flask, request, json
from waitress import serve
import __main__

__main__.ModelParameters = ModelParameters

app = Flask(__name__)
your_rest_server_port = 5000


@app.get("/api/v1/models/predict/<ticker>")
def get_full_stock_info(ticker: str):
    prediction = get_prediction(ticker)
    return str(prediction)

@app.get("/test")
def get():
    return "Hello world"


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=your_rest_server_port)
