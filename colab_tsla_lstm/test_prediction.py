import os
import numpy as np
import pandas as pd
from lstm_predict import predict_price, plot_prices, get_accuracy
from lstm_model import build_lstm, preprocess_data


#---------------------------------------------------
# Test: predict the future price
file_name = 'TSLA.csv'
scaler_name = 'minmax'
n_steps = 20
lookup_step = 1
test_size = 0.2 # 20% of data will be used for test

#---------------------------------------------------
# preprocess data from the file downloaded
# and get training data
data = preprocess_data(file_name, scaler_name, n_steps, lookup_step, test_size, False)
print('data.keys() =', data.keys())
feats = data['feats']
n_feats = len(feats)
print(n_feats)

# get the model and load weights
model = build_lstm(n_steps, n_feats)
ticker = 'TSLA' ; loss = 'mse'
model_name = f"{ticker}-{loss}-seq-{n_steps}-step-{lookup_step}"
model_path = os.path.join('results', model_name) + '.h5'
model.load_weights(model_path)

# evaluate the model
mse, mae = model.evaluate(data['x_test'], data['y_test'], verbose=0)
# calculate the mean absolute error (inverse scaling)
scale = data['scaler']
mean_absolute_error = scale['adjclose'].inverse_transform([[mae]])[0][0]
print("Mean Absolute Error:", mean_absolute_error)


future_price = predict_price(model, data, n_steps)
print(f"Future price {lookup_step} days ahead is {future_price:.2f}$")

#---------------------------------------------------
# prediction with the prices of last n_days
n_days = 100
plot_prices(model, data, n_days)


#---------------------------------------------------
# i day(s) ahead prediction of price
for i in [1, 10, 30, 60]:
    print(str(i) + " day(s) ahead :", "Accuracy Score:", get_accuracy(model, data, i))






