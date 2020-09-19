#from lstm_model import build_lstm, preprocess_data
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import accuracy_score


def predict_price(model, data, n_steps):
    """predict a price one day ahead based on 
    the last sequence of seq_len = n_steps from data"""
    seq_last = data['seq_last'][-n_steps:] #(seq_len, n_feats)
    seq_last = np.expand_dims(seq_last, axis=0)# reshape for model.predict
    # get the prediction (scaled from 0 to 1)
    pred = model.predict(seq_last)
    # get scaler for each column
    scale = data['scaler']
    # get the price (by inverting the scaling) and [0][0] returns number
    predicted_price = scale['adjclose'].inverse_transform(pred)[0][0]
    return predicted_price

def plot_prices(model, data, n_days):
    """plot true prices and predicted prices during n_days"""
    # change the type of dates in df for plot 
    df = data['df']
    dates = df[df.columns[0]][-n_days:] # object type
    dates = pd.to_datetime(dates)       # transformed to use in plot

    y_test = data['y_test']
    x_test = data['x_test'] # (510, 20, 6) : 510 sequences of each len = 20, y_pred expected in (510,)
    y_pred = model.predict(x_test)
    # To convert y_pred back to the price in original scale  
    scale = data['scaler']
    y_test = np.squeeze(scale['adjclose'].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(scale['adjclose'].inverse_transform(y_pred))
    # last n_days days
    plt.figure(figsize=(10,5))
    ax = plt.gca()
    plt.plot(dates, y_test[-n_days:])
    plt.plot(dates, y_pred[-n_days:], c='orange')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.gcf().autofmt_xdate() # rotation at angle
    ax.set_title(df['ticker'][0])
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

def get_accuracy(model, data, lookup_step):
    y_test = data["y_test"]
    x_test = data['x_test']
    y_pred = model.predict(x_test)
    scale = data['scaler']
    y_test = np.squeeze(scale["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(scale["adjclose"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-lookup_step], y_pred[lookup_step:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-lookup_step], y_test[lookup_step:]))
    return accuracy_score(y_test, y_pred)

