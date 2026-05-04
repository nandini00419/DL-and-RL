import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Reads the CSV file containing historical stock data
# Loads it into a Pandas DataFrame called data
data = pd.read_csv("data/stock_data.csv")

#Keeps only the Close price column
#Removes all rows
#Result in clean closing price data
data = data[['Close']].dropna()

#Converts the DataFrame column into a NumPy array
#Shape becomes -number_of_days, 1
#Needed because ML models work with NumPy arrays, not DataFrames
close_prices = data.values

#Creates a scaler that rescales values to the range 0 - 1
#LSTMs perform much better with normalized data
scaler = MinMaxScaler(feature_range=(0, 1))

#Learns the min and max of prices
#Transforms prices into values between 0 and 1
#Eg:
#$100 - 0.0
#$200 - 1.0
scaled_data = scaler.fit_transform(close_prices)

# X: input sequences
# y: target values
X, y = [], []

#Each prediction uses the previous 60 days
#Easy choice for stock time-series
window_size = 60

#Loops over the dataset starting at index 60
#Ensures we have 60 past values for each prediction
for i in range(window_size, len(scaled_data)):

#Takes the previous 60 scaled prices
#Adds them as one input sample
#Shape of one sample: (60,)
    X.append(scaled_data[i-window_size:i, 0])

#The price right after the 60-day window
#This is what the LSTM must learn to predict
    y.append(scaled_data[i, 0])

#Converts Python lists into NumPy arrays
#Required for TensorFlow/Keras
X = np.array(X)
y = np.array(y)

#Reshapes input for LSTM:
#(samples, time_steps, features)
#Here:
#features = 1-only Close price
X = X.reshape((X.shape[0], X.shape[1], 1))

# Uses 80% of data for training
# Remaining 20% for testing
split = int(0.8 * len(X))

#Training inputs
#Testing inputs
X_train, X_test = X[:split], X[split:]

#Training targets
#Testing targets
y_train, y_test = y[:split], y[split:]

#Creates a stacked neural network
#Layers are executed top -bottom
#First LSTM layer with 50 memory units
#return_sequences=True:
#Needed because another LSTM comes next
#input_shape = (60, 1) → 60 days, 1 feature
##Second LSTM layer
#Outputs one vector but not sequences
#Extracts higher-level temporal patterns
#Fully connected layer
#Outputs one predicted price
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

#Adam optimizer-adaptive learning rate
#MSE- penalizes large prediction errors
model.compile(optimizer='adam', loss='mean_squared_error')

# Trains the model
# epochs=10: dataset seen 10 times
# batch_size=32: updates weights every 32 samples
# verbose=1: shows progress bar
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Predicts scaled prices on unseen data
predicted_scaled = model.predict(X_test)

#Converts predictions back to real price values
#.flatten() makes it 1D
predicted = scaler.inverse_transform(predicted_scaled).flatten()

#Converts actual test values back to original prices
#Makes comparison possible
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 8. REINFORCEMENT LEARNING
# RL parameters
#actions = [0, 1, 2]
#Action space:
#0 = Hold
#1 = Buy
#2 = Sell
actions = [0, 1, 2]

#Learning rate
#How fast the agent updates Q-values
alpha = 0.1

#Discount factor
#Future rewards are worth 95% of current rewards
gamma = 0.95

#Exploration rate
#10% random actions
epsilon = 0.1

# Q-table:
# 3 states × 3 actions
# Stores expected rewards
Q = np.zeros((3, len(actions)))

#Converts prices into a discrete trend state
def get_state(pred, curr):

    #Predicted price higher - Uptrend
    if pred > curr:
        return 2

    #Predicted lower - Downtrend
    elif pred < curr:
        return 0

    #No change - Neutral
    else:
        return 1

#Total accumulated profit
portfolio = 0

#0 = no stock
#1 = holding stock
position = 0

#Tracks profit history
profits = []

#Iterates through time
for t in range(1, len(predicted)):

    #Determines current market state
    state = get_state(predicted[t], actual[t])

    # Exploration: random action
    if np.random.rand() < epsilon:
        action = np.random.choice(actions)

        #Exploitation: best known action
    else:
        action = np.argmax(Q[state])

#Default reward
    reward = 0

    # Buy only if not already holding stock
    if action == 1 and position == 0:

        #Stores purchase price
        #Marks position as open
        buy_price = actual[t]
        position = 1

    # Sell only if holding stock
    elif action == 2 and position == 1:

        #Profit = sell price − buy price
        reward = actual[t] - buy_price

        #Update total profit
        # Close position
        portfolio += reward
        profits.append(portfolio)
        position = 0

    # Determines next market state
    next_state = get_state(predicted[t], actual[t])

    #Updates expected reward for -state, action
    Q[state, action] += alpha * (
        reward + gamma * np.min(Q[next_state]) - Q[state, action]
    )

print("\n💰 Total RL Trading Profit:", portfolio)


# Average squared error
mse = mean_squared_error(actual, predicted)

#Root Mean Squared Error
rmse = np.sqrt(mse)

#Average absolute error
mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

print("\n📊 LSTM Evaluation Metrics:")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R2   : {r2:.4f}")

# Creates plot canvas
plt.figure(figsize=(12,6))
#Plots real vs predicted prices
plt.plot(actual, label="Actual Price")
plt.plot(predicted, label="Predicted Price")
#Displays the graph
plt.legend()
plt.title("LSTM Stock Price Prediction")
plt.show()

# RL Profit Curve
plt.figure(figsize=(10,5))
plt.plot(profits)
#Displays trading performance
plt.title("Reinforcement Learning Trading Profit")
plt.xlabel("Trades")
plt.ylabel("Profit")
plt.show()



