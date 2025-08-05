import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# get data from yfinance and drop unnecessary columns
ticker = yf.Ticker("^IXIC")
hist = ticker.history(period="1y")
df = hist.drop(columns=["Dividends", "Stock Splits", "Volume"])  # drop columns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(df.head())

# normalize the data to [0, 1]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

# create sequences of length 'sequence_length' with targets
def create_sequences(data, sequence_length):
    x = []
    y = []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])  # target is the next value after the sequence
    return np.array(x), np.array(y)

# sigmoid and tanh activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

#forward pass for one sequence
def forward_one_sequence(sequence_input, W_input, W_hidden, bias_gates, W_output, bias_output):
    hidden_state = np.zeros((1, hidden_size))  # hidden state
    cell_state = np.zeros((1, hidden_size))    # cell state

    for t in range(sequence_length):
        input_t = sequence_input[t].reshape(1, input_size)
        gates = input_t @ W_input + hidden_state @ W_hidden + bias_gates

        input_gate = sigmoid(gates[:, :hidden_size])
        forget_gate = sigmoid(gates[:, hidden_size:2*hidden_size])
        output_gate = sigmoid(gates[:, 2*hidden_size:3*hidden_size])
        candidate = tanh(gates[:, 3*hidden_size:])

        cell_state = forget_gate * cell_state + input_gate * candidate
        hidden_state = output_gate * tanh(cell_state)

    prediction = hidden_state @ W_output + bias_output  # final prediction
    return prediction[0, 0], hidden_state  # return both prediction and last hidden state

# prepping input & target data
close_prices = scaled_df["Close"].values
sequence_length = 10  # num of days used for prediction

x, y = create_sequences(close_prices, sequence_length)

# lstm parameters
input_size = 1          # for closing price
hidden_size = 5         # num of LSTM units

# lstm weights
W_input = np.random.randn(input_size, 4 * hidden_size) * 0.1  # input to gates
W_hidden = np.random.randn(hidden_size, 4 * hidden_size) * 0.1  # hidden to gates
bias_gates = np.zeros((1, 4 * hidden_size))  # gate biases

# manual step-by-step forward pass for first sequence
first_sequence = x[0].reshape(sequence_length, input_size)

hidden_state = np.zeros((1, hidden_size))
cell_state = np.zeros((1, hidden_size))

for t in range(sequence_length):
    input_t = first_sequence[t].reshape(1, input_size)
    gates = input_t @ W_input + hidden_state @ W_hidden + bias_gates

    input_gate = sigmoid(gates[:, :hidden_size])
    forget_gate = sigmoid(gates[:, hidden_size:2*hidden_size])
    output_gate = sigmoid(gates[:, 2*hidden_size:3*hidden_size])
    candidate = tanh(gates[:, 3*hidden_size:])

    cell_state = forget_gate * cell_state + input_gate * candidate
    hidden_state = output_gate * tanh(cell_state)

    # output for each time step
    # print(f"\nTime step {t + 1}")
    # print("input_t:", input_t)
    # print("gates:", gates)
    # print("input gate:", input_gate)
    # print("forget gate:", forget_gate)
    # print("output gate:", output_gate)
    # print("candidate:", candidate)
    # print("updated cell state:", cell_state)
    # print("updated hidden state:", hidden_state)

# output layer parameters
W_output = np.random.randn(hidden_size, 1) * 0.1
bias_output = np.zeros((1, 1))

# final prediction from last hidden state
final_prediction = hidden_state @ W_output + bias_output
print("\npredicted next value (normalized):", final_prediction[0, 0])

# training loop, updating output layer
learning_rate = 0.01
epochs = 10

for epoch in range(epochs):
    mse_total = 0

    for i in range(len(x)):
        sequence_input = x[i].reshape(sequence_length, input_size)
        true_value = y[i]

        # use the function for forward pass
        predicted_value, final_hidden = forward_one_sequence(sequence_input, W_input, W_hidden, bias_gates, W_output, bias_output)

        loss = (predicted_value - true_value) ** 2
        mse_total += loss

        # output layer gradient and update
        dL_dy = 2 * (predicted_value - true_value) #derivative of mse losswith respect to models output
        dL_dy = np.array([[dL_dy]])  # change to 2d array

        dW_output = final_hidden.T @ dL_dy # gradient of loss with respect to output weight matrix
        db_output = dL_dy #gradient of loss with respect to output bias

        W_output -= learning_rate * dW_output
        bias_output -= learning_rate * db_output

    print(f"epoch {epoch+1}, MSE: {mse_total / len(x)}")
