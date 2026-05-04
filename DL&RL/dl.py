import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load a dataset
data = pd.read_csv("data/stock_data.csv")

# Kept only the close price and remove rest of the cloumm
data = data[['Close']].dropna()
# extract the values from your data object in panda seires data frame ,
# close_prices becomes a NumPy array containing stock prices.
close_prices = data.values

print("Total data points:", len(close_prices))
print("NaNs after cleaning:", np.isnan(close_prices).sum())

# Creates a Min-Max scaler, It scales values to the range 0 to 1, which helps LSTM(Long short term memory) train
scaler = MinMaxScaler(feature_range=(0, 1))

#Learns the min & max from close_prices, Transforms prices into scaled values between 0 and 1.
scaled_data = scaler.fit_transform(close_prices)

# X will store input sequences (past prices).
# y will store target values (next price).
X, y = [], []

# Each and every input sample uses 60 previous days to predict the next day.
window_size = 60

#Loop starts at index 60, Ensures we always have 60 past values available.
for i in range(window_size, len(scaled_data)):

    # Adds them as one training sample, 0 means we’re using the first column of Close price.
    X.append(scaled_data[i-window_size:i, 0])

    #target value is the price at time i, This will  the model will learn to predict.
    y.append(scaled_data[i, 0])

# convert list into numpy array
X = np.array(X)
y = np.array(y)

#it reshapes the data here
# sample = number of sequences
# timesteps = 60
# features = 1-Close price
X = X.reshape((X.shape[0], X.shape[1], 1))

print("X shape:", X.shape)
print("y shape:", y.shape)

# Uses 80% of data for training,
# Remaining 20% for testing.
split = int(0.8 * len(X))

#Splits data with the chronologically which is important for time series,
# No shuffling it avoids the  data leakage.
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#Initializes a sequential neural network.
model = Sequential()
#first LSTM layer with 50 neurons, return_sequences= True-outputs full sequences for another LSTM,
# Input shape - 60 timesteps, 1 feature
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#Second LSTM layer.
#Returns only the final output.
model.add(LSTM(50))
#Fully connected layer,
# Outputs one predicted price.
model.add(Dense(1))

#Optimizer: Adam ,Loss: Mean Squared Error-common for regression.
model.compile(optimizer='adam', loss='mean_squared_error')
# dsiplay the ,model summary with there architetcure
model.summary()

# Trains model for 10 epochs.
# Uses 32 samples per batch.
# verbose=1 shows progress.
print("\nTraining the model...")
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    verbose=1
)

# Generates predictions in scaled form 0–1.
predicted_scaled = model.predict(X_test)

# Inverse scaling
#Converts predictions back to actual price values.
predicted = scaler.inverse_transform(predicted_scaled)
#Converts true test prices back to original scale.
#reshape(-1, 1) required for the scaler
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Check NaNs value
print("NaNs in actual:", np.isnan(actual).sum())
print("NaNs in predicted:", np.isnan(predicted).sum())

# Average squared prediction error.
mse = mean_squared_error(actual, predicted)
#Square root of MSE-error in price units.
rmse = np.sqrt(mse)
#Average absolute difference between predicted and actual prices.
mae = mean_absolute_error(actual, predicted)
#Measures how well predictions explain variance in prices.
r2 = r2_score(actual, predicted)

print("\n📊 Model Evaluation Metrics:")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R2   : {r2:.4f}")

# Time Series Plot,
# Plots real vs predicted prices over time.
plt.figure(figsize=(12,6))
plt.plot(actual, label='Actual Price')
plt.plot(predicted, label='Predicted Price')
#Adds labels and displays the plot
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

# Regression Scatter Plot
#Shows relationship between actual and predicted prices.
plt.figure(figsize=(6,6))
sns.scatterplot(x=actual.flatten(), y=predicted.flatten())
#Red dashed line = perfect prediction line.
plt.plot(
    [actual.min(), actual.max()],
    [actual.min(), actual.max()],
    'r--'
)

#Helps visually assess prediction accuracy.
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Regression")
plt.show()
