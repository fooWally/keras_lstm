import pickle
import matplotlib.pyplot as plt

f = open('history.pkl', 'rb')
hist = pickle.load(f)
f.close()



fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,6))
ax1.plot(hist['loss'], 'y', label='train loss')
ax1.plot(hist['val_loss'], 'r', label='val loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('mse loss')
ax1.legend(loc='upper left')

ax2.plot(hist['mae'], 'b', label='train mae')
ax2.plot(hist['val_mae'], 'k', label='val mae')
ax2.set_xlabel('epoch')
ax2.set_ylabel('mean average error')
ax2.legend(loc='upper right')

plt.show()
