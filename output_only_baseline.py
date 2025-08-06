#!/usr/bin/env python3
"""
Updated comparison with better hyperparameter tuning and analysis
"""

import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Get and prepare data (same as main script)
ticker = yf.Ticker("^IXIC")
hist = ticker.history(period="1y")
df = hist[["Open", "High", "Low", "Close"]]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(x), np.array(y)

# Activation functions (same as main)
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(np.clip(x, -500, 500))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t * t

# Prepare data
close_prices = scaled_df["Close"].values
sequence_length = 10
x, y = create_sequences(close_prices, sequence_length)

# Split into train/test
split_idx = int(0.8 * len(x))
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# LSTM parameters
input_size = 1
hidden_size = 5

print("=== Enhanced LSTM Training Analysis ===")
print(f"Total dataset size: {len(x)} sequences")
print(f"Training set: {len(x_train)} sequences")
print(f"Test set: {len(x_test)} sequences")
print(f"Sequence length: {sequence_length}")
print(f"Hidden size: {hidden_size}")
print()

# Initialize weights
np.random.seed(42)
W_input_init = np.random.randn(input_size, 4 * hidden_size) * 0.1
W_hidden_init = np.random.randn(hidden_size, 4 * hidden_size) * 0.1
bias_gates_init = np.zeros((1, 4 * hidden_size))
W_output_init = np.random.randn(hidden_size, 1) * 0.1
bias_output_init = np.zeros((1, 1))

def forward_simple(sequence_input, W_input, W_hidden, bias_gates, W_output, bias_output):
    """Simple forward pass (output-only training)"""
    hidden_state = np.zeros((1, hidden_size))
    cell_state = np.zeros((1, hidden_size))

    for t in range(sequence_length):
        input_t = sequence_input[t].reshape(1, input_size)
        gates = input_t @ W_input + hidden_state @ W_hidden + bias_gates

        input_gate = sigmoid(gates[:, :hidden_size])
        forget_gate = sigmoid(gates[:, hidden_size:2*hidden_size])
        output_gate = sigmoid(gates[:, 2*hidden_size:3*hidden_size])
        candidate = tanh(gates[:, 3*hidden_size:])

        cell_state = forget_gate * cell_state + input_gate * candidate
        hidden_state = output_gate * tanh(cell_state)

    prediction = hidden_state @ W_output + bias_output
    return prediction[0, 0], hidden_state

def evaluate_model(x_data, y_data, weights, forward_fn):
    """Evaluate model on dataset"""
    total_mse = 0
    total_mae = 0
    
    for i in range(len(x_data)):
        sequence_input = x_data[i].reshape(sequence_length, input_size)
        true_value = y_data[i]
        
        if forward_fn == forward_simple:
            predicted_value, _ = forward_fn(sequence_input, *weights)
        else:
            predicted_value, _ = forward_fn(sequence_input, *weights)
        
        mse = (predicted_value - true_value) ** 2
        mae = abs(predicted_value - true_value)
        total_mse += mse
        total_mae += mae
    
    return total_mse / len(x_data), total_mae / len(x_data)

def train_output_only(epochs=50, lr=0.01):
    """Train only output layer"""
    W_input = W_input_init.copy()
    W_hidden = W_hidden_init.copy()
    bias_gates = bias_gates_init.copy()
    W_output = W_output_init.copy()
    bias_output = bias_output_init.copy()
    
    train_losses = []
    test_losses = []
    
    print(f"Training OUTPUT-ONLY with lr={lr}...")
    
    for epoch in range(epochs):
        # Training
        mse_total = 0
        for i in range(len(x_train)):
            sequence_input = x_train[i].reshape(sequence_length, input_size)
            true_value = y_train[i]
            
            predicted_value, final_hidden = forward_simple(sequence_input, W_input, W_hidden, bias_gates, W_output, bias_output)
            
            loss = (predicted_value - true_value) ** 2
            mse_total += loss
            
            # Output layer gradients only
            dL_dy = 2 * (predicted_value - true_value)
            dL_dy = np.array([[dL_dy]])
            
            dW_output = final_hidden.T @ dL_dy
            db_output = dL_dy
            
            W_output -= lr * dW_output
            bias_output -= lr * db_output
        
        train_mse = mse_total / len(x_train)
        train_losses.append(train_mse)
        
        # Test evaluation
        test_mse, _ = evaluate_model(x_test, y_test, (W_input, W_hidden, bias_gates, W_output, bias_output), forward_simple)
        test_losses.append(test_mse)
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}: Train MSE = {train_mse:.6f}, Test MSE = {test_mse:.6f}")
    
    return train_losses, test_losses, (W_input, W_hidden, bias_gates, W_output, bias_output)

# Run output-only training
train_losses_output, test_losses_output, weights_output = train_output_only()

print("\n=== RESULTS ANALYSIS ===")
final_train_mse = train_losses_output[-1]
final_test_mse = test_losses_output[-1]

print(f"Output-only training:")
print(f"  Final training MSE: {final_train_mse:.6f}")
print(f"  Final test MSE: {final_test_mse:.6f}")
print(f"  Test RMSE: {np.sqrt(final_test_mse):.6f}")

# Test predictions on a few samples
print("\n=== SAMPLE PREDICTIONS ===")
for i in range(min(5, len(x_test))):
    sequence_input = x_test[i].reshape(sequence_length, input_size)
    true_value = y_test[i]
    
    pred, _ = forward_simple(sequence_input, *weights_output)
    
    # Denormalize for interpretation
    true_denorm = scaler.inverse_transform([[0, 0, 0, true_value]])[0, 3]
    pred_denorm = scaler.inverse_transform([[0, 0, 0, pred]])[0, 3]
    
    print(f"Test sample {i+1}: True=${true_denorm:.2f}, Predicted=${pred_denorm:.2f}, Error=${abs(true_denorm - pred_denorm):.2f}")

print(f"\nOutput-only LSTM training completed successfully!")
print(f"The model shows the baseline performance without full LSTM backpropagation.")
print(f"For comparison, run the main.py script which implements full backpropagation.")

# Show the key differences
print(f"\n=== KEY IMPLEMENTATION DIFFERENCES ===")
print("1. OUTPUT-ONLY TRAINING (this script):")
print("   - Only updates W_output and bias_output weights")
print("   - LSTM internal weights (W_input, W_hidden, bias_gates) remain fixed")
print("   - Faster training but limited learning capacity")
print()
print("2. FULL BACKPROPAGATION (main.py):")
print("   - Updates ALL weights through backpropagation through time (BPTT)")
print("   - Computes gradients for all LSTM gates and hidden states")
print("   - More complex but can learn richer temporal patterns")
print("   - Includes gradient clipping for numerical stability")
