import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# get data from yfinance and drop unnecessary columns
ticker = yf.Ticker("^IXIC")
hist = ticker.history(period="5y")
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
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])  # target is the next value after the sequence
    return np.array(x), np.array(y)


# sigmoid and tanh activation functions with derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # clip to prevent overflow


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(np.clip(x, -500, 500))  # clip to prevent overflow


def tanh_derivative(x):
    t = tanh(x)
    return 1 - t * t


# Forward pass for one sequence with state storage for back propagation
def forward_one_sequence(sequence_input, W_input, W_hidden, bias_gates, W_output, bias_output):
    # Initialize the states and storage for back propagation
    hidden_states = np.zeros((sequence_length + 1, hidden_size))
    cell_states = np.zeros((sequence_length + 1, hidden_size))

    # Store gate values and intermediate computations for back propagation
    input_gates = np.zeros((sequence_length, hidden_size))
    forget_gates = np.zeros((sequence_length, hidden_size))
    output_gates = np.zeros((sequence_length, hidden_size))
    candidates = np.zeros((sequence_length, hidden_size))
    gate_inputs = np.zeros((sequence_length, 4 * hidden_size))
    tanh_cell_states = np.zeros((sequence_length, hidden_size))

    for t in range(sequence_length):
        input_t = sequence_input[t].reshape(1, input_size)

        # 1. Compute the gates
        # 2. Apply activations
        # 3. Update cell and hidden states
        gates = input_t @ W_input + hidden_states[t].reshape(1, -1) @ W_hidden + bias_gates
        gate_inputs[t] = gates.flatten()

        input_gate = sigmoid(gates[:, :hidden_size])
        forget_gate = sigmoid(gates[:, hidden_size:2 * hidden_size])
        output_gate = sigmoid(gates[:, 2 * hidden_size:3 * hidden_size])
        candidate = tanh(gates[:, 3 * hidden_size:])

        input_gates[t] = input_gate.flatten()
        forget_gates[t] = forget_gate.flatten()
        output_gates[t] = output_gate.flatten()
        candidates[t] = candidate.flatten()

        cell_states[t + 1] = forget_gate.flatten() * cell_states[t] + input_gate.flatten() * candidate.flatten()
        tanh_cell_states[t] = tanh(cell_states[t + 1])
        hidden_states[t + 1] = output_gates[t] * tanh_cell_states[t]

    prediction = hidden_states[-1].reshape(1, -1) @ W_output + bias_output

    # Store all intermediate values for backprop
    forward_cache = {
        'hidden_states': hidden_states,
        'cell_states': cell_states,
        'input_gates': input_gates,
        'forget_gates': forget_gates,
        'output_gates': output_gates,
        'candidates': candidates,
        'gate_inputs': gate_inputs,
        'tanh_cell_states': tanh_cell_states,
        'sequence_input': sequence_input
    }

    return prediction[0, 0], forward_cache


# Backpropagation through LSTM
def backward_one_sequence(forward_cache, dL_dy, W_input, W_hidden, W_output):
    hidden_states = forward_cache['hidden_states']
    cell_states = forward_cache['cell_states']
    input_gates = forward_cache['input_gates']
    forget_gates = forward_cache['forget_gates']
    output_gates = forward_cache['output_gates']
    candidates = forward_cache['candidates']
    gate_inputs = forward_cache['gate_inputs']
    tanh_cell_states = forward_cache['tanh_cell_states']
    sequence_input = forward_cache['sequence_input']

    # Initialize gradients
    dW_input = np.zeros_like(W_input)
    dW_hidden = np.zeros_like(W_hidden)
    db_gates = np.zeros((1, 4 * hidden_size))

    dh_next = (dL_dy.reshape(1, -1) @ W_output.T).flatten()
    dc_next = np.zeros(hidden_size)

    # backprop through time
    for t in reversed(range(sequence_length)):
        # current hidden and cell state gradients
        dh = dh_next
        dc = dc_next

        # gradients w.r.t. output gate
        do = dh * tanh_cell_states[t]
        dc += dh * output_gates[t] * tanh_derivative(cell_states[t + 1])

        # gradients w.r.t. cell state from previous timestep
        dc_prev = dc * forget_gates[t]

        # gradients w.r.t. forget gate
        df = dc * cell_states[t]

        # gradients w.r.t. input gate and candidate
        di = dc * candidates[t]
        dg = dc * input_gates[t]

        # gradients w.r.t. gate inputs (before activation)
        do_input = do * sigmoid_derivative(gate_inputs[t, 2 * hidden_size:3 * hidden_size])
        df_input = df * sigmoid_derivative(gate_inputs[t, hidden_size:2 * hidden_size])
        di_input = di * sigmoid_derivative(gate_inputs[t, :hidden_size])
        dg_input = dg * tanh_derivative(gate_inputs[t, 3 * hidden_size:])

        # concatenate gate gradients
        dgate = np.concatenate([di_input, df_input, do_input, dg_input])

        # gradients w.r.t. weights and biases
        input_t = sequence_input[t].reshape(-1, 1)
        hidden_prev = hidden_states[t].reshape(-1, 1)

        dW_input += input_t @ dgate.reshape(1, -1)
        dW_hidden += hidden_prev @ dgate.reshape(1, -1)
        db_gates += dgate.reshape(1, -1)

        # gradients w.r.t. previous hidden state
        dh_next = (dgate.reshape(1, -1) @ W_hidden.T).flatten()
        dc_next = dc_prev

    return dW_input, dW_hidden, db_gates


# prepping input & target data
close_prices = scaled_df["Close"].values
sequence_length = 20  # num of days used for prediction
x, y = create_sequences(close_prices, sequence_length)
# 80/20 split
split_idx = int(0.8 * len(x))
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# lstm parameters
input_size = 1  # for closing price
hidden_size = 16  # num of LSTM units

# lstm weights
W_input = np.random.randn(input_size, 4 * hidden_size) * 0.1  # input to gates
W_hidden = np.random.randn(hidden_size, 4 * hidden_size) * 0.1  # hidden to gates
bias_gates = np.zeros((1, 4 * hidden_size))  # gate biases

# output layer parameters
W_output = np.random.randn(hidden_size, 1) * 0.1
bias_output = np.zeros((1, 1))

# test initial prediction
print("Testing initial model before training:")
first_sequence = x[0].reshape(sequence_length, input_size)
initial_prediction, _ = forward_one_sequence(first_sequence, W_input, W_hidden, bias_gates, W_output, bias_output)
print(f"Initial predicted next value (normalized): {initial_prediction:.6f}")
print(f"Actual next value (normalized): {y[0]:.6f}")
print(f"Initial error: {abs(initial_prediction - y[0]):.6f}")
print()

# training loop with full backpropagation
learning_rate = 0.01  # reduced learning rate for stability
epochs = 100
gradient_clip_threshold = 1.0  # gradient clipping for numerical stability

print("Training LSTM with full backpropagation...")
print(f"Dataset size: {len(x)} sequences")
print(f"Sequence length: {sequence_length}")
print(f"Hidden size: {hidden_size}")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {epochs}")
print(f"Gradient clipping: {gradient_clip_threshold}")


def clip_gradients(grad, threshold):
    grad_norm = np.linalg.norm(grad)
    if grad_norm > threshold:
        grad = grad * (threshold / grad_norm)
    return grad


for epoch in range(epochs):
    mse_total_train = 0.0

    # zero accumulators
    dW_input_total = np.zeros_like(W_input)
    dW_hidden_total = np.zeros_like(W_hidden)
    db_gates_total = np.zeros_like(bias_gates)
    dW_output_total = np.zeros_like(W_output)
    db_output_total = np.zeros_like(bias_output)

    for i in range(len(x_train)):
        sequence_input = x_train[i].reshape(sequence_length, input_size)
        true_value = y_train[i]
        pred, cache = forward_one_sequence(sequence_input, W_input, W_hidden, bias_gates, W_output, bias_output)
        loss = (pred - true_value) ** 2
        mse_total_train += loss

        dL_dy = 2 * (pred - true_value)
        dL_dy = np.array([[dL_dy]])

        final_hidden = cache['hidden_states'][-1].reshape(1, -1)
        dW_output = final_hidden.T @ dL_dy
        db_output = dL_dy

        dW_input, dW_hidden, db_gates = backward_one_sequence(cache, dL_dy, W_input, W_hidden, W_output)

        # clip
        dW_input = clip_gradients(dW_input, gradient_clip_threshold)
        dW_hidden = clip_gradients(dW_hidden, gradient_clip_threshold)
        db_gates = clip_gradients(db_gates, gradient_clip_threshold)
        dW_output = clip_gradients(dW_output, gradient_clip_threshold)
        db_output = clip_gradients(db_output, gradient_clip_threshold)

        # accumulate
        dW_input_total += dW_input
        dW_hidden_total += dW_hidden
        db_gates_total += db_gates
        dW_output_total += dW_output
        db_output_total += db_output

    n = len(x_train)
    W_input  -= learning_rate * dW_input_total  / n
    W_hidden -= learning_rate * dW_hidden_total / n
    bias_gates -= learning_rate * db_gates_total / n
    W_output -= learning_rate * dW_output_total / n
    bias_output -= learning_rate * db_output_total / n

    train_mse = mse_total_train / len(x_train)

    mse_total_test, mae_total_test = 0.0, 0.0
    for i in range(len(x_test)):
        sequence_input = x_test[i].reshape(sequence_length, input_size)
        true_value = y_test[i]
        pred, _ = forward_one_sequence(sequence_input, W_input, W_hidden, bias_gates, W_output, bias_output)
        mse_total_test += (pred - true_value) ** 2
        mae_total_test += abs(pred - true_value)

    test_mse = mse_total_test / len(x_test)
    test_rmse = np.sqrt(test_mse)
    test_mae = mae_total_test / len(x_test)

    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_mse:.6f} | "
              f"Test MSE: {test_mse:.6f}, Test RMSE: {test_rmse:.6f}, Test MAE: {test_mae:.6f}")

print("\nTraining completed!")

# test the model on a few predictions (test set only)
print("\nTesting predictions:")
for i in range(min(5, len(x_test))):
    sequence_input = x_test[i].reshape(sequence_length, input_size)
    true_value = y_test[i]
    predicted_value, _ = forward_one_sequence(sequence_input, W_input, W_hidden, bias_gates, W_output, bias_output)
    true_denorm = scaler.inverse_transform([[0,0,0,true_value]])[0,3]
    pred_denorm = scaler.inverse_transform([[0,0,0,predicted_value]])[0,3]
    # print(f"Sample {i+1}: True={true_denorm:.2f}, Predicted={pred_denorm:.2f}, Error={abs(true_denorm - pred_denorm):.2f}")

print("\nFinal evaluation (test set):")
total_mse = total_mae = 0.0
for i in range(len(x_test)):
    sequence_input = x_test[i].reshape(sequence_length, input_size)
    true_value = y_test[i]
    predicted_value, _ = forward_one_sequence(sequence_input, W_input, W_hidden, bias_gates, W_output, bias_output)
    total_mse += (predicted_value - true_value) ** 2
    total_mae += abs(predicted_value - true_value)

final_mse = total_mse / len(x_test)
print(f"Final MSE: {final_mse:.6f}")
print(f"Final MAE: {total_mae / len(x_test):.6f}")
print(f"Final RMSE: {np.sqrt(final_mse):.6f}")