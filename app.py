from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
import datetime
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dateutil import parser
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from google.cloud import storage

app = Flask(__name__)
CORS(app)


BUCKET_NAME = os.environ.get('BUCKET_NAME', 'lstm_bkt_spmasst')

BASE_IMAGE_PATH = f"https://storage.googleapis.com/{BUCKET_NAME}/"

LOCAL_IMAGE_PATH = "static/images/"






client = storage.Client()

# GitHub Auth
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "your_token")
HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

END_DATE = datetime.datetime.utcnow()
START_DATE = END_DATE - datetime.timedelta(days=60)


def fetch_github_data(owner, repo, endpoint, params=None):
    results, page = [], 1
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/{endpoint}"
        p = {'per_page': 100, 'page': page}
        if params: p.update(params)
        resp = requests.get(url, headers=HEADERS, params=p)
        if resp.status_code != 200: break
        data = resp.json()
        if not data: break
        results.extend(data)
        page += 1
    return results


def df_from_dates(dates):
    parsed = [parser.parse(d).date() for d in dates]
    df = pd.DataFrame({'date': parsed})
    df = df.groupby('date').size().reset_index(name='count')
    df = df.set_index('date').asfreq('D', fill_value=0)
    return df


def upload_to_gcs(local_filename, gcs_filename):
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_filename)
    blob.upload_from_filename(filename=local_filename)


def save_plot(df, forecast=None, title="Plot", filename="output.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['count'], label='Actual')
    if forecast is not None:
        future_idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30)
        plt.plot(future_idx, forecast, '--', label='Forecast')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    path = os.path.join(LOCAL_IMAGE_PATH, filename)
    plt.savefig(path)
    upload_to_gcs(path, filename)
    return BASE_IMAGE_PATH + filename


@app.route("/api/forecast", methods=["POST"])
def forecast_all():
    body = request.get_json()
    repo = body.get("repo")
    metric = body.get("metric")  # 'issues', 'pulls', 'commits'

    owner, repo_name = repo.split("/")

    if metric == "issues":
        data = fetch_github_data(owner, repo_name, "issues", {'state': 'all', 'since': START_DATE.isoformat()})
        created_dates = [i['created_at'] for i in data if 'pull_request' not in i]
    elif metric == "pulls":
        data = fetch_github_data(owner, repo_name, "pulls", {'state': 'all'})
        created_dates = [i['created_at'] for i in data]
    elif metric == "commits":
        data = fetch_github_data(owner, repo_name, "commits", {'since': START_DATE.isoformat()})
        created_dates = [i['commit']['author']['date'] for i in data]
    else:
        return jsonify({"error": "Invalid metric type"}), 400

    df = df_from_dates(created_dates)

    responses = {}

    # Prophet Forecast
    prophet_df = df.reset_index().rename(columns={'date': 'ds', 'count': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    prophet_path = f"forecast_prophet_{metric}_{repo_name}.png"
    fig = model.plot(forecast)
    fig.savefig(os.path.join(LOCAL_IMAGE_PATH, prophet_path))
    upload_to_gcs(os.path.join(LOCAL_IMAGE_PATH, prophet_path), prophet_path)
    responses['prophet'] = BASE_IMAGE_PATH + prophet_path

    # ARIMA Forecast
    arima_model = ARIMA(df['count'], order=(1, 1, 1))
    arima_result = arima_model.fit()
    arima_forecast = arima_result.forecast(30)
    arima_url = save_plot(df, arima_forecast, f"ARIMA Forecast - {metric}", f"forecast_arima_{metric}_{repo_name}.png")
    responses['arima'] = arima_url

    # LSTM Forecast
    series = df['count'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series)
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[i])
        y.append(data[i + 1])
    X = np.array(X).reshape((len(X), 1, 1))
    y = np.array(y)

    model = Sequential([LSTM(50, activation='relu', input_shape=(1, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, verbose=0)
    input_seq = data[-1].reshape((1, 1, 1))
    preds = []
    for _ in range(30):
        p = model.predict(input_seq)
        preds.append(p[0, 0])
        input_seq = np.array(p).reshape((1, 1, 1))
    lstm_forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    lstm_url = save_plot(df, lstm_forecast, f"LSTM Forecast - {metric}", f"forecast_lstm_{metric}_{repo_name}.png")
    responses['lstm'] = lstm_url

    return jsonify(responses)


@app.route("/")
def index():
    return "GitHub Forecast Microservice: Supports LSTM, ARIMA, Prophet \U0001F680"


# Run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
