import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout
import matplotlib.pyplot as plt

df = web.DataReader('ASHOKLEY.NS',data_source='yahoo',start='2002-07-01',end='2020-03-23')
data = df.filter(['Open'])
dataset = data.values
training_data_len = math.ceil( len(dataset) * 0.8)
print (training_data_len)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len,:]
x_train = []
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)
#building model
model = Sequential()
model.add(LSTM(100,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(100,return_sequences=False))
#model.add(Dropout(0.5))
#model.add(LSTM(50,return_sequences=False))
#model.add(LSTM(25,return_sequences=False))
model.add(Dense(50))
#model.add(Dense(25))
#model.add(Dense(5))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=1,epochs=5)
model.save('Opening.h5')
test_data =scaled_data[training_data_len-60 : ,:]
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
print(x_test.shape)
x_test = np.reshape(x_test , (x_test.shape[0],x_test.shape[1],1))
prediction = model.predict(x_test)
prediction = scaler.inverse_transform(prediction)
rmse = np.sqrt(np.mean(prediction-y_test)**2)
print(rmse)
train = data[:training_data_len]
valid = data[training_data_len:]
valid['prediction'] = prediction
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('date',fontsize=18)
plt.ylabel('close price')
plt.plot(train['Open'])
plt.plot(valid[['Open','prediction']])
plt.show()
print(valid)

