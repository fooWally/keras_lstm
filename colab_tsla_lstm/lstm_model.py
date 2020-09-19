import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def build_lstm(n_steps, n_feats, n_layers=2):
    #--------------------------------
    #units : number of LSTM neurons
    #n_layers : number of layers
    #40% dropout
    drop = 0.4
    #--------------------------------
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # 1st layer
            model.add(LSTM(units=32, return_sequences=True, input_shape=(n_steps, n_feats)))
        elif i == n_layers - 1:
            # last layer
            model.add(LSTM(units=16, return_sequences=False))
        else:
            # hidden layers
            model.add(LSTM(units=16, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(drop))
    model.add(Dense(16, activation="linear"))
    model.add(Dense(8, activation="linear"))
    model.add(Dense(1, activation="linear"))

    start = time.time()
    model.compile(loss='mse', metrics=['mae'], optimizer='rmsprop')
    #print("Compilation Time : ", time.time() - start)
    return model

def preprocess_data(file_name, scaler_name, n_steps, lookup_step, test_size, shuffle=True):
    """df  : dataframe instance of panda
    scaler : import StandardScaler/MinMaxScaler and pass it to arg """
    data = {}
    df = pd.read_csv(file_name)
    
    # Select 6 feats out of df.keys()
    feats = df.keys()[1:7] # remove Date from dataframe and ticker
    print('feats =', feats)
    lookup_step = 1
    n_steps = 20

    data['feats'] = feats
    data['df'] = df.copy()
    #----------------------------------------------------
    # Each f needs its own scaler:
    # Save each scaler for later prediction in inverse_tranformation
    col_scaler = {}
    for f in feats:
        if scaler_name == 'standard' :
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        df[f] = scaler.fit_transform(np.expand_dims(df[f].values, axis=1))
        col_scaler[f] = scaler
    data['scaler'] = col_scaler

    #----------------------------------------------------
    # Add future column to df and shift by lookup_step and leave NANs
    # Hence there are NAN in df, which we want to drop later
    try :
        df['future'] = df['Adj Close'].shift(-lookup_step)
    except:
         df['future'] = df['adjclose'].shift(-lookup_step)
    # Drop rows with NaNs in df
    df.dropna(inplace=True)

    #----------------------------------------------------
    # slice data by n_steps of window_size and 
    # put sequences of _n_steps in data
    seq_n_step = deque(maxlen=n_steps) # e.g n_steps = 20 ; window_size
    seq_last = np.array(df[feats].tail(lookup_step))

    X, y = [], []
    for feat_vals, target in zip(df[feats].values, df['future'].values):
        # everytime appending one, push one out keeps its len 20
        seq_n_step.append(feat_vals)
        # when data_n_steps is full, append it to data_all
        if len(seq_n_step) == n_steps:
            X.append(np.array(seq_n_step))
            y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    seq_last = list(seq_n_step) + list(seq_last)
    seq_last = np.array(seq_last)

    # save to data
    data['seq_last'] = seq_last
    data['x_train'], data['x_test'], data['y_train'], data['y_test'] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    return data
