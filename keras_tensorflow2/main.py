from sklearn.datasets import load_diabetes
from sklearn.utils import Bunch
from numpy import ndarray
from tensorflow.compat.v1.random import set_random_seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
from datetime import datetime


datasets: Bunch = load_diabetes()
inputs: ndarray = datasets.data
targets: ndarray = datasets.target.reshape(-1, 1)
print(f'inputs: {inputs.shape}, targets: {targets.shape}')
set_random_seed(1)


optimizer = Adam(learning_rate = 0.001)
model = Sequential([
    Dense(64, input_dim = 10, activation = relu),
    Dense(128, activation = relu),
    Dense(64, activation = relu),
    Dense(1, activation = None)
])
model.compile(loss = mean_squared_error, optimizer = optimizer, metrics = ['accuracy'])
print(model.summary())


start = datetime.now()
model.fit(inputs, targets, epochs = 100, verbose = 1)
print(f'Time taken: {datetime.now() - start}')