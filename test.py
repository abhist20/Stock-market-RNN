import math
import pandas_datareader as web
import numpy as np
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
scaler = MinMaxScaler(feature_range=(0,1))
model=keras.models.load_model('opening.h5')
df1 = web.DataReader('ASHOKLEY.NS',data_source='yahoo',start='2019-01-01',end='2020-03-11')
new_df = df1.filter(['Open'])
last_60_days = new_df[-60:].values
last_60_scaled = scaler.fit_transform(last_60_days)
x_test = []
x_test.append(last_60_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)
print(pred)