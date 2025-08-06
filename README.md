## Project Topic - RNN/LSTM for Time Series Prediction

### Requirements:
- You have to **code** one or more algorithms from the technique that you have studied on your chosen dataset. 
  You are free to use pre-processing and evaluations libraries, but the main algorithm should be coded by you.
- A log file of your experiments (i.e. runs) and hyper-parameters should be maintained and submitted.

### LSTM Implementation Features:
- **Custom LSTM Architecture**: Hand-coded LSTM with forget gates, input gates, output gates, and candidate values
- **Full Backpropagation Through Time (BPTT)**: Complete gradient computation through all LSTM gates and hidden states
- **Time Series Prediction**: Stock market data (NASDAQ Composite Index) prediction using 10-day sequences
- **Data Preprocessing**: MinMax normalization and sequence generation for time series
- **Gradient-based Training**: Mini-batch gradient descent with accumulated gradients

### Experiment Log:

| Experiment Number | Parameter Chosen | Results |
|-------------------|------------------|---------|
| 1 | **Full BPTT LSTM**:<br>Hidden Size = 5<br>Sequence Length = 10<br>Learning Rate = 0.001<br>Epochs = 50<br>Batch Processing = Full Dataset<br>Error Function = MSE<br>Gradient Clipping = 1.0 | Dataset Size = 240 sequences<br>Input Features = Close Price (normalized)<br>Train/Test Split = None (full dataset training)<br>Final MSE = 0.339101<br>Final RMSE = 0.582323<br>Final MAE = 0.546815<br>Convergence = Steady decrease |
| 2 | **Output-Only Baseline**:<br>Hidden Size = 5<br>Sequence Length = 10<br>Learning Rate = 0.01<br>Epochs = 50<br>LSTM Weights = Fixed<br>Only Output Layer Training | Dataset Size = 240 sequences<br>Train/Test Split = 80:20<br>Training MSE = 0.024034<br>Test MSE = 0.113493<br>Test RMSE = 0.336888<br>Faster convergence but limited capacity |
| 3 | **Architecture Comparison**:<br>Full BPTT vs Output-Only<br>Same dataset and conditions<br>Different learning rates optimized | **Full BPTT**: More complex patterns<br>**Output-Only**: Faster training<br>**Key Finding**: Output-only shows better immediate results on this dataset due to simpler optimization landscape, but full BPTT has greater learning potential |
_______________

### Technical Implementation Details:

#### Forward Pass:
- **Input Processing**: Each timestep processes normalized closing price
- **Gate Computations**: Four gates (input, forget, output, candidate) computed simultaneously
- **State Updates**: Cell state and hidden state updated sequentially through time
- **Output Generation**: Final hidden state mapped to scalar prediction

#### Backward Pass:
- **Gradient Flow**: Gradients propagated backward through time (BPTT)
- **Gate Derivatives**: Computed using chain rule for sigmoid and tanh activations
- **Weight Updates**: All weight matrices (W_input, W_hidden, W_output) and biases updated
- **Gradient Accumulation**: Gradients accumulated across all sequences before weight updates

#### Key Features:
- **Numerical Stability**: Gradient clipping to prevent overflow/underflow
- **Memory Efficiency**: Forward cache stores intermediate values for backprop
- **Complete Implementation**: No external LSTM libraries used - fully original code
