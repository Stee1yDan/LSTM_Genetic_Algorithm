from model_prediction_handler import get_prediction, ModelParameters
from flask import Flask, request, json
import py_eureka_client.eureka_client as eureka_client
from waitress import serve
import __main__

__main__.ModelParameters = ModelParameters

app = Flask(__name__)
your_rest_server_port = 5000

eureka_client.init(eureka_server="http://localhost:8761/eureka",
                   instance_host="0.0.0.0",
                   app_name="stock-prediction-service",
                   instance_port=your_rest_server_port)

@app.get("/api/v1/models/predict/<ticker>")
def get_full_stock_info(ticker: str):
    prediction = get_prediction(ticker)
    return str(prediction)

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=your_rest_server_port)
