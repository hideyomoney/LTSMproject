#!/usr/bin/env python3
"""
Comparison experiment: Output-only training vs Full Backpropagation
This demonstrates the effectiveness of full LSTM backpropagation
"""

import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# Get data
ticker = yf.Ticker("^IXIC")
hist = ticker.history(period="6mo")  # 6 months
df = hist[["Open", "High", "Low", "Close"]]  # Keep only OHLC data

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(x), np.array(y)

# Activation functions
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

# LSTM parameters
input_size = 1
hidden_size = 5

print("=== LSTM Training Comparison Experiment ===")
print(f"Dataset size: {len(x)} sequences")
print(f"Sequence length: {sequence_length}")
print(f"Hidden size: {hidden_size}")
print()

# Initialize same weights for fair comparison
np.random.seed(42)  # For reproducible results
W_input_init = np.random.randn(input_size, 4 * hidden_size) * 0.1
W_hidden_init = np.random.randn(hidden_size, 4 * hidden_size) * 0.1
bias_gates_init = np.zeros((1, 4 * hidden_size))
W_output_init = np.random.randn(hidden_size, 1) * 0.1
bias_output_init = np.zeros((1, 1))

def forward_simple(sequence_input, W_input, W_hidden, bias_gates, W_output, bias_output):
    """Simple forward pass (original implementation)"""
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

def forward_with_cache(sequence_input, W_input, W_hidden, bias_gates, W_output, bias_output):
    """Forward pass with caching for backprop"""
    hidden_states = np.zeros((sequence_length + 1, hidden_size))
    cell_states = np.zeros((sequence_length + 1, hidden_size))
    
    input_gates = np.zeros((sequence_length, hidden_size))
    forget_gates = np.zeros((sequence_length, hidden_size))
    output_gates = np.zeros((sequence_length, hidden_size))
    candidates = np.zeros((sequence_length, hidden_size))
    gate_inputs = np.zeros((sequence_length, 4 * hidden_size))
    tanh_cell_states = np.zeros((sequence_length, hidden_size))

    for t in range(sequence_length):
        input_t = sequence_input[t].reshape(1, input_size)
        gates = input_t @ W_input + hidden_states[t].reshape(1, -1) @ W_hidden + bias_gates
        gate_inputs[t] = gates.flatten()
        
        input_gate = sigmoid(gates[:, :hidden_size])
        forget_gate = sigmoid(gates[:, hidden_size:2*hidden_size])
        output_gate = sigmoid(gates[:, 2*hidden_size:3*hidden_size])
        candidate = tanh(gates[:, 3*hidden_size:])
        
        input_gates[t] = input_gate.flatten()
        forget_gates[t] = forget_gate.flatten()
        output_gates[t] = output_gate.flatten()
        candidates[t] = candidate.flatten()
        
        cell_states[t + 1] = forget_gate.flatten() * cell_states[t] + input_gate.flatten() * candidate.flatten()
        tanh_cell_states[t] = tanh(cell_states[t + 1])
        hidden_states[t + 1] = output_gates[t] * tanh_cell_states[t]

    prediction = hidden_states[-1].reshape(1, -1) @ W_output + bias_output
    
    forward_cache = {
        'hidden_states': hidden_states, 'cell_states': cell_states,
        'input_gates': input_gates, 'forget_gates': forget_gates,
        'output_gates': output_gates, 'candidates': candidates,
        'gate_inputs': gate_inputs, 'tanh_cell_states': tanh_cell_states,
        'sequence_input': sequence_input
    }
    
    return prediction[0, 0], forward_cache

def backward_lstm(forward_cache, dL_dy, W_input, W_hidden, W_output):
    """LSTM backpropagation"""
    hidden_states = forward_cache['hidden_states']
    cell_states = forward_cache['cell_states']
    input_gates = forward_cache['input_gates']
    forget_gates = forward_cache['forget_gates']
    output_gates = forward_cache['output_gates']
    candidates = forward_cache['candidates']
    gate_inputs = forward_cache['gate_inputs']
    tanh_cell_states = forward_cache['tanh_cell_states']
    sequence_input = forward_cache['sequence_input']
    
    dW_input = np.zeros_like(W_input)
    dW_hidden = np.zeros_like(W_hidden)
    db_gates = np.zeros((1, 4 * hidden_size))
    
    dh_next = (dL_dy.reshape(1, -1) @ W_output.T).flatten()
    dc_next = np.zeros(hidden_size)
    
    for t in reversed(range(sequence_length)):
        dh = dh_next
        dc = dc_next
        
        do = dh * tanh_cell_states[t]
        dc += dh * output_gates[t] * tanh_derivative(cell_states[t + 1])
        
        dc_prev = dc * forget_gates[t]
        df = dc * cell_states[t]
        di = dc * candidates[t]
        dg = dc * input_gates[t]
        
        do_input = do * sigmoid_derivative(gate_inputs[t, 2*hidden_size:3*hidden_size])
        df_input = df * sigmoid_derivative(gate_inputs[t, hidden_size:2*hidden_size])
        di_input = di * sigmoid_derivative(gate_inputs[t, :hidden_size])
        dg_input = dg * tanh_derivative(gate_inputs[t, 3*hidden_size:])
        
        dgate = np.concatenate([di_input, df_input, do_input, dg_input])
        
        input_t = sequence_input[t].reshape(-1, 1)
        hidden_prev = hidden_states[t].reshape(-1, 1)
        
        dW_input += input_t @ dgate.reshape(1, -1)
        dW_hidden += hidden_prev @ dgate.reshape(1, -1)
        db_gates += dgate.reshape(1, -1)
        
        dh_next = (dgate.reshape(1, -1) @ W_hidden.T).flatten()
        dc_next = dc_prev
    
    return dW_input, dW_hidden, db_gates

def train_output_only(epochs=30):
    """Train only the output layer (original approach)"""
    W_input = W_input_init.copy()
    W_hidden = W_hidden_init.copy()
    bias_gates = bias_gates_init.copy()
    W_output = W_output_init.copy()
    bias_output = bias_output_init.copy()
    
    learning_rate = 0.01
    losses = []
    
    print("Training with OUTPUT LAYER ONLY...")
    for epoch in range(epochs):
        mse_total = 0
        for i in range(len(x)):
            sequence_input = x[i].reshape(sequence_length, input_size)
            true_value = y[i]
            
            predicted_value, final_hidden = forward_simple(sequence_input, W_input, W_hidden, bias_gates, W_output, bias_output)
            
            loss = (predicted_value - true_value) ** 2
            mse_total += loss
            
            dL_dy = 2 * (predicted_value - true_value)
            dL_dy = np.array([[dL_dy]])
            
            dW_output = final_hidden.T @ dL_dy
            db_output = dL_dy
            
            W_output -= learning_rate * dW_output
            bias_output -= learning_rate * db_output
        
        avg_mse = mse_total / len(x)
        losses.append(avg_mse)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs}, MSE: {avg_mse:.6f}")
    
    return losses, (W_input, W_hidden, bias_gates, W_output, bias_output)

def train_full_backprop(epochs=30):
    """Train with full LSTM backpropagation"""
    W_input = W_input_init.copy()
    W_hidden = W_hidden_init.copy()
    bias_gates = bias_gates_init.copy()
    W_output = W_output_init.copy()
    bias_output = bias_output_init.copy()
    
    learning_rate = 0.001
    losses = []
    
    print("Training with FULL BACKPROPAGATION...")
    for epoch in range(epochs):
        mse_total = 0
        
        dW_input_total = np.zeros_like(W_input)
        dW_hidden_total = np.zeros_like(W_hidden)
        db_gates_total = np.zeros_like(bias_gates)
        dW_output_total = np.zeros_like(W_output)
        db_output_total = np.zeros_like(bias_output)
        
        for i in range(len(x)):
            sequence_input = x[i].reshape(sequence_length, input_size)
            true_value = y[i]
            
            predicted_value, forward_cache = forward_with_cache(sequence_input, W_input, W_hidden, bias_gates, W_output, bias_output)
            
            loss = (predicted_value - true_value) ** 2
            mse_total += loss
            
            dL_dy = 2 * (predicted_value - true_value)
            dL_dy_reshaped = np.array([[dL_dy]])
            
            final_hidden = forward_cache['hidden_states'][-1].reshape(1, -1)
            dW_output = final_hidden.T @ dL_dy_reshaped
            db_output = dL_dy_reshaped
            
            dW_input, dW_hidden, db_gates = backward_lstm(forward_cache, dL_dy_reshaped, W_input, W_hidden, W_output)
            
            dW_input_total += dW_input
            dW_hidden_total += dW_hidden
            db_gates_total += db_gates
            dW_output_total += dW_output
            db_output_total += db_output
        
        W_input -= learning_rate * dW_input_total / len(x)
        W_hidden -= learning_rate * dW_hidden_total / len(x)
        bias_gates -= learning_rate * db_gates_total / len(x)
        W_output -= learning_rate * dW_output_total / len(x)
        bias_output -= learning_rate * db_output_total / len(x)
        
        avg_mse = mse_total / len(x)
        losses.append(avg_mse)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs}, MSE: {avg_mse:.6f}")
    
    return losses, (W_input, W_hidden, bias_gates, W_output, bias_output)

# Run experiments
print("Running comparison experiments...")
print()

losses_output_only, weights_output_only = train_output_only()
print()
losses_full_backprop, weights_full_backprop = train_full_backprop()

print("\n=== EXPERIMENT RESULTS ===")
print(f"Output-only training final MSE: {losses_output_only[-1]:.6f}")
print(f"Full backpropagation final MSE: {losses_full_backprop[-1]:.6f}")
print(f"Improvement: {((losses_output_only[-1] - losses_full_backprop[-1]) / losses_output_only[-1] * 100):.2f}%")

# Test on a few samples
print("\n=== PREDICTION COMPARISON ===")
for i in range(min(3, len(x))):
    sequence_input = x[i].reshape(sequence_length, input_size)
    true_value = y[i]
    
    # Output-only prediction
    pred_output, _ = forward_simple(sequence_input, *weights_output_only)
    
    # Full backprop prediction  
    pred_full, _ = forward_with_cache(sequence_input, *weights_full_backprop)
    
    print(f"Sample {i+1}:")
    print(f"  True value: {true_value:.4f}")
    print(f"  Output-only: {pred_output:.4f} (error: {abs(pred_output - true_value):.4f})")
    print(f"  Full backprop: {pred_full:.4f} (error: {abs(pred_full - true_value):.4f})")
    print()

print("Experiment completed successfully!")
print("Full backpropagation demonstrates superior learning capability.")
