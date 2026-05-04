import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import random

# Load a dataset
data = pd.read_csv("data/stock_data.csv")
# Kept only the close price and remove rest of the cloumm
# extract the values from your data object in panda seires data frame ,
# close_prices becomes a NumPy array containing stock prices.
close_prices = data[['Close']].dropna().values

print("Total data points:", len(close_prices))
print("NaNs after cleaning:", np.isnan(close_prices).sum())

# Creates a Min-Max scaler, It scales values to the range 0 to 1,
# which helps LSTM(Long short term memory) train
scaler = MinMaxScaler(feature_range=(0, 1))
#Learns the min & max from close_prices,
# Transforms prices into scaled values between 0 and 1.
scaled_data = scaler.fit_transform(close_prices)

# Each and every input sample uses 60 previous days to predict the next day.
window_size = 60

# X will store input sequences.
# y will store target values.
X, y = [], []

#Loop starts at index 60, Ensures we always have 60 past values available.
for i in range(window_size, len(scaled_data)):
    # Adds them as one training sample,
    # 0 means we’re using the first column of Close price.
    X.append(scaled_data[i - window_size:i, 0])
    # target value is the price at time i,
    # This will  the model will learn to predict
    y.append(scaled_data[i, 0])

# Adds them as one training sample,
# 0 means we’re using the first column of Close price.
#target value is the price at time i,
# This will  the model will learn to predict.
X, y = np.array(X), np.array(y)
#it reshapes the data here
# sample = number of sequences
# timesteps = 60
# features = 1-Close price
X = X.reshape((X.shape[0], X.shape[1], 1))

# Uses 80% of data for training,
# Remaining 20% for testing.
split = int(0.8 * len(X))

#Splits data with the chronologically which is important for time series,
# No shuffling it avoids the  data leakage.
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# first LSTM layer with 50 neurons, return_sequences= True-outputs full sequences for another LSTM,
#Initializes a sequential neural network.
#first LSTM layer with 50 neurons, return_sequences= True-outputs full sequences for another LSTM,
# Input shape - 60 timesteps, 1 feature
#Second LSTM layer.
#Returns only the final output.
#Fully connected layer,
# Outputs one predicted price.
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
    LSTM(50),
    Dense(1)
])
#Optimizer: Adam ,Loss: Mean Squared Error-common for regression.
lstm_model.compile(optimizer='adam', loss='mse')

# Trains model for 10 epochs.
# Uses 32 samples per batch.
# verbose=1 shows progress.
lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# Predicts prices on test data -still scaled
predicted_scaled = lstm_model.predict(X_test)
#Converts predictions back to real prices
predicted_prices = scaler.inverse_transform(predicted_scaled).flatten()
#Converts true prices back to real scale
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()


# 3. DEEP Q-NETWORK (DQN) AGENT
#purpose decide weather to buy ,sell.
#this will define a class with a name hybridDQN agent where there are many objects
#which are defined by the functions
#Defines a reinforcement learning agent
class HybridDQNAgent:
    # This is the constructor of the class.It initializes everything the DQN
    # agent needs state_size-number of inputs-features in the state
    # action_size-number of possible actions ie.Hold, Buy, Sell
    def __init__(self, state_size, action_size):

        # stores how many values describe the environment state
        #example : state = [predicted_price, actual_price, inventory]
           #state_size = 3
        self.state_size = state_size

        #Stores how many actions the agent can take
       # eg: 0 = Hold
       #1 = Buy
       #2 = Sell
        #action_size = 3
        self.action_size = action_size

        # Creates a double-ended queue (fast add/remove)
        # Stores past experiences as
        #eg: (state, action, reward, next_state, done)
        #maxlen=2000-when memory is full oldest experiences are automatically removed
        self.memory = deque(maxlen=2000)

        #The agent cares about future rewards
        #gamma = 0 → only immediate reward matters
        #gamma = 1 → future rewards matter as much as immediate ones
        #gamma = 0.95 → future rewards matter, but slightly less
        self.gamma = 0.95

        #epsilon = 1.0 -100% random actions
        #Encourages exploration
        self.epsilon = 1.0

        #Sets a lower bound
        # Agent will always explore at least 1% of the time
        self.epsilon_min = 0.01

        # Controls how fast exploration decreases
        self.epsilon_decay = 0.995

        #Calls another method that builds the Q-network
        #This network: Takes the state as input
        # Outputs Q - values for each action
        self.model = self._build_model()

    #This builds the neural network that approximates Q-values
    #In DQN, this network replaces a Q-table
    def _build_model(self):
        #A simple feed-forward neural network
        #nput_dim = state_size (3 values: prediction, price, inventory)
       #24 neurons
        #ReLU activation → handles non-linearity
        #deeper relationships between state features
        #One neuron per action
        #Outputs Q-values, not probabilities
        #linear because Q-values can be any real number
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        #Loss: Mean Squared Error
        #Optimizer: Adam
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

#Decides what action to take in the current state
    def act(self, state):
        # Generates a random number between 0 and 1
        # If it’s less than epsilon-explore
        if np.random.rand() <= self.epsilon:

            #Pick a random action -Hold / Buy / Sell
            return random.randrange(self.action_size)
        #Network predicts Q-values for all actions
        act_values = self.model.predict(state, verbose=0)
        #Chooses the action with the highest Q-value
        return np.argmax(act_values[0])
         #Trains the agent using past experiences


    def train_replay(self, batch_size):
        #Randomly selects experiences from memory
        #Prevents learning from correlated data
        minibatch = random.sample(self.memory, batch_size)

        #loop through expereiences
        for state, action, reward, next_state, done in minibatch:
            #Start with immediate reward
            target = reward
            #If episode is not over, include future reward
            if not done:
                # Bellman Equation : How good the action , considering and what to next
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
           #Current Q-values for the state
            target_f = self.model.predict(state, verbose=0)
           #Update only the Q-value of the action taken
            #Other action values remain unchanged
            target_f[0][action] = target

            #Train the network so predicted Q-values move closer to target
            self.model.fit(state, target_f, epochs=1, verbose=0)

            #Gradually shift from exploration → exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# TRADING SIMULATION
#State contains:
#LSTM predicted price
#Current market price
#Inventory size
state_size = 3

#Creates the DQN agent
# 0=Hold, 1=Buy, 2=Sell
agent = HybridDQNAgent(state_size, action_size=3)
batch_size = 32

#Tracks open positions and profitTracks open positions and profit
inventory = []
total_profit = 0

#Small negative reward to discourage inactivity
portfolio_history = []
living_penalty = -0.01

#Simulates trading day by day
for t in range(len(predicted_prices) - 1):
    # Current market state
    state = np.array([[predicted_prices[t], actual_prices[t], len(inventory)]])
    #Agent decides: Hold / Buy / Sell
    action = agent.act(state)

#Default penalty if nothing happens
    reward = living_penalty

    # Buy stock-store purchase price
    if action == 1:
        inventory.append(actual_prices[t])
        #Sell only if stock is owned
    elif action == 2 and len(inventory) > 0:
        buy_price = inventory.pop(0)
        #Reward = profit
        # Profit accumulates
        reward = actual_prices[t] - buy_price  # Profit is the reward
        total_profit += reward

    #Save experience for replay learning
    next_state = np.array([[predicted_prices[t + 1], actual_prices[t + 1], len(inventory)]])
    agent.memory.append((state, action, reward, next_state, False))
    portfolio_history.append(total_profit)

    # Trains the agent every 10 steps
    # Avoids overfitting to recent data
    if t > batch_size and t % 10 == 0:
        agent.train_replay(batch_size)

print(f"Total Profit from Hybrid RL: {total_profit:.2f}")

# 5. VISUALIZATION
plt.figure(figsize=(12, 5))
plt.plot(portfolio_history, label="DQN Profit Curve", color='green')
plt.title("Hybrid RL Performance (LSTM + DQN)")
plt.xlabel("Days")
plt.ylabel("Cumulative Profit")
plt.legend()
plt.show()