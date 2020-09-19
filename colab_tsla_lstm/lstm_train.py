import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from lstm_model import build_lstm, preprocess_data 
import pickle, time

#---------------------------------------------
#
#   DATA PREPROCESS
#
# preprocess data from the file downloaded
# and get training data
file_name = 'TSLA.csv'
scaler_name = 'minmax' #'standard'
n_steps = 20  # window size
lookup_step = 1
test_size = 0.2 # 20% of data will be used for test

data = preprocess_data(file_name, scaler_name, n_steps, lookup_step, test_size)
print('data.keys() =', data.keys())
x_train, y_train = data['x_train'], data['y_train'] 
x_test, y_test = data['x_test'], data['y_test']
print('x_train.shape =', x_train.shape)
print('x_test.shape =', x_test.shape)
feats = data['feats']
n_feats = len(feats)
print(n_feats)

#---------------------------------------------
#   create training/validation data set
k = 3; # imagine 3-fold cross-validation
idx = np.random.choice(len(y_train),int(len(y_train)/k))
print('len(idx) =', len(idx))
#print(val_idx)
x_val = x_train[idx]
y_val = y_train[idx]
print('x_val.shape =', x_val.shape)
print('y_val.shape =', x_val.shape)

idx_all = np.arange(len(y_train))
idx_left = set(idx_all) - set(idx)
idx_left = np.array(list(idx_left))

x_train = x_train[idx_left]
y_train = y_train[idx_left]
print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)

#--------------------------------------------
#
#   TRAIN
#
# Construct Lstm model
# Train and save callbacks : 
#date_now = time.strftime("%Y-%m-%d")
ticker = 'TSLA' ; loss = 'mse'
model_name = f"{ticker}-{loss}-seq-{n_steps}-step-{lookup_step}"

model = build_lstm(n_steps, n_feats)
model.summary()

# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")
    

checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
early_stopping = EarlyStopping(patience = 10)
hist = model.fit(x_train, y_train, batch_size=128, epochs=100,
                validation_data=(x_val, y_val),
                callbacks=[checkpointer, tensorboard, early_stopping],verbose=1)

model.save(os.path.join("results", model_name) + ".h5")
results = model.evaluate(x_test, y_test, batch_size=128)

# save hist to history.pkl
f = open('history.pkl', 'wb')
pickle.dump(hist.history, f)
f.close()


# plot loss and mae
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,6))
ax1.plot(hist.history['loss'], 'y', label='train loss')
ax1.plot(hist.history['val_loss'], 'r', label='val loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('mse loss')
ax1.legend(loc='upper left')

ax2.plot(hist.history['mae'], 'b', label='train mae')
ax2.plot(hist.history['val_mae'], 'k', label='val mae')
ax2.set_xlabel('epoch')
ax2.set_ylabel('mean average error')
ax2.legend(loc='upper right')

plt.show()



