import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#Load Data 

ticker = ('Amazon', "AMZN")

start = dt.datetime(2010, 1, 1)
end = dt.datetime(2020, 12, 31)

dataset = pdr.DataReader(ticker[1], 'yahoo', start, end)

#Prepare Data

#1. data normalization

predicted_days = 26

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset['Adj Close'].values.reshape(-1, 1))

x_train = []
y_train = []

for x in range(predicted_days, len(scaled_data)):
    x_train.append(scaled_data[x-predicted_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#2. Build Model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

#Test the model accuracy on existing data

#3. load test data
test_start = dt.datetime(2021, 1, 1)
test_end = dt.datetime.now()

test_data = pdr.DataReader(ticker[1], 'yahoo', test_start, test_end)

actual_prices = test_data['Adj Close'].values

total_dataset = pd.concat((dataset['Adj Close'], test_data['Adj Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - predicted_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

#4. make predictions on test data
x_test = []

for x in range(predicted_days, len(model_inputs)):
    x_test.append(model_inputs[x-predicted_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#5. Plot the results

fig, ax = plt.subplots()
ax.set(xlabel="Date",
       ylabel="Price",
       title=f'{ticker[0]} Stock Price')
ax.plot(actual_prices, color='blue', label=f'Actual {ticker[0]} Price')
ax.plot(predicted_prices, color='red', linestyle='dashed', label=f'Predicted {ticker[0]} Price')
plt.legend()
plt.show()

real_data = [model_inputs[len(model_inputs)+1-predicted_days:len(model_inputs)+1,0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data,(real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Tomorrow's price is predicted to be: {prediction}")
