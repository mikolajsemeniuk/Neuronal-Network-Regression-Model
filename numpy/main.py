from sklearn.datasets import load_diabetes
from sklearn.utils import Bunch
import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split


datasets: Bunch = load_diabetes()
inputs: ndarray = datasets.data
targets: ndarray = datasets.target.reshape(-1, 1)
print(f'inputs: {inputs.shape}, targets: {targets.shape}')


inputs_train: ndarray
inputs_test: ndarray
targets_train: ndarray
targets_test: ndarray
inputs_train, inputs_test, targets_train, targets_test = \
    train_test_split(inputs, targets, test_size = 0.3, random_state = 2021)


class Layer_Dense:
      
    def __init__(self, in_dimensions: int, out_dimensions: int) -> None:
        self.weights: ndarray = np.random.randn(in_dimensions, out_dimensions)
        self.biases: ndarray = np.zeros((1, out_dimensions))
      
    def forward(self, inputs: ndarray) -> None:
        self.inputs: ndarray = inputs
        self.output: ndarray = inputs @ self.weights + self.biases
      
    def backward(self, dvalues: ndarray) -> None:
        self.dweights: ndarray = self.inputs.T @ dvalues
        self.dbiases: ndarray = np.sum(dvalues, axis = 0, keepdims=True)
        self.dinputs: ndarray = dvalues @ self.weights.T


class Activation_ReLU:

    def forward(self, inputs: ndarray) -> None:
        self.inputs: ndarray = inputs
        self.output: ndarray = np.maximum(0, inputs)
    
    def backward(self, dvalues) -> None:
        self.dinputs: ndarray = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Linear:
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()


class Loss_MeanSquaredError:

    def calculate(self, output, y):
        return np.mean(self.forward(output, y))
    
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


class Loss_MeanAbsoluteError:

    def calculate(self, output, y):
        return np.mean(self.forward(output, y))
    
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


class Loss_BinaryCrossentropy:
    def calculate(self, output, y):
        return np.mean(self.forward(output, y))

    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(sample_losses, axis = -1)

    def backward(self, dvalues, y_true):
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / len(dvalues[0])
        self.dinputs = self.dinputs / len(dvalues)


class Optimizer_SGD:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


class Optimizer_Adam:

    def __init__(self, learning_rate=0.001, epsilon=1e-08,
        beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
            
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)


dense1 = Layer_Dense(10, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 128)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(128, 64)
activation3 = Activation_ReLU()
dense4 = Layer_Dense(64, 1)
activation4 = Activation_Linear()

loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_Adam(learning_rate=0.005)
accuracy_precision = np.std(targets_train) / 250

for epoch in range(1000):
    
    dense1.forward(inputs_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    dense4.forward(activation3.output)
    activation4.forward(dense4.output)
    
    loss = loss_function.calculate(activation4.output, targets_train)
    
    predictions = activation4.output
    accuracy = np.mean(np.absolute(predictions - targets_train) < accuracy_precision)
    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {accuracy:.4f}, loss: {loss:.4f}')
    
    loss_function.backward(activation4.output, targets_train)
    activation4.backward(loss_function.dinputs)
    dense4.backward(activation4.dinputs)
    activation3.backward(dense4.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.update_params(dense4)