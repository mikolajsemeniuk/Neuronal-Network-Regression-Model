from sklearn.datasets import load_diabetes
from sklearn.utils import Bunch
from numpy import ndarray
from torch import manual_seed, Tensor, FloatTensor
from torch.nn import Sequential, Linear, LeakyReLU, MSELoss
from torch.optim import Adam


datasets: Bunch = load_diabetes()
inputs: ndarray = datasets.data
targets: ndarray = datasets.target.reshape(-1, 1)
print(f'inputs: {inputs.shape}, targets: {targets.shape}')


manual_seed(1)
X: Tensor = FloatTensor(inputs)
y: Tensor = FloatTensor(targets)


model = Sequential(
    Linear(10, 64),
    LeakyReLU(),
    Linear(64, 128),
    LeakyReLU(),
    Linear(128, 64),
    LeakyReLU(),
    Linear(64, 1)
)


epochs: int = 1000
optimizer = Adam(model.parameters(), lr=0.001)
loss_function = MSELoss()


for epoch in range(epochs):
    optimizer.zero_grad()

    predictions: Tensor = \
        model(X)
    
    loss: Tensor = \
        loss_function(predictions, y)
    
    print(f'epoch: {epoch + 1}, loss: {loss}')

    loss.backward()
    optimizer.step()