# predicting stock price using Recurrent Neural Network (RNN) - LSTM

#------------------------------------------------------------------
## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the Keras libraries and packages for RNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

#------------------------------------------------------------------
# Step 1 - Data Preprocessing

## Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# consider only one single stock column to train on
training_set = dataset_train.iloc[:, 1:2].values

## Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

## train with 60 timesteps to predict 1 output
X_train = []
y_train = []
# total entries = 1258
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

## Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#------------------------------------------------------------------
# Step 2 - Building and Training the RNN

## Initialising the RNN
rnn = Sequential()

## Layer 1: first LSTM layer and Dropout regularisation (to avoid overfitting)
rnn.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
rnn.add(Dropout(0.2))

## Layer 2: second LSTM layer and Dropout regularisation
rnn.add(LSTM(units = 50, return_sequences = True))
rnn.add(Dropout(0.2))

## Layer 3: third LSTM layer and Dropout regularisation
rnn.add(LSTM(units = 50, return_sequences = True))
rnn.add(Dropout(0.2))

## Layer 4: fourth LSTM layer and Dropout regularisation
rnn.add(LSTM(units = 50)) # no return_sequences = True, because this is the last LSTM layer
rnn.add(Dropout(0.2))

## Layer 5: the output layer
rnn.add(Dense(units = 1)) # units = 1 for output layer as we are predicting a value (regression problem)

## Compiling the RNN
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error') # MSE error because it's a regression problem

## training the RNN
rnn.fit(X_train, y_train, epochs = 250, batch_size = 32)

#------------------------------------------------------------------
# Step 3 - Evaluation and visualising the results

## Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

## Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = rnn.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

## Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()