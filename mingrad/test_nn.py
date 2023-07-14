import numpy as np
from engine import Value
from nn import Linear, ReLU, Sequential, MAE, SGD

# linear model
model = Sequential(
    Linear(2, 4),
    ReLU(),
    Linear(4, 2)
)

# optimizer
optim = SGD(model.parameters(), lr=0.01)

# random input data
x = Value(np.random.randn(4, 2))  # bs = 4
print(f"x = {x.data}")

# random targets
targets = Value(np.random.randn(4, 2))

# fp
out = model(x)
print(f"out (before SGD) = {out.data}")

# loss
mae = MAE()
loss = mae(out, targets)
print(f"loss (before SGD step) = {loss.data}")

# bp
loss.backward()

# optimizer step
optim.step()

# fp after SGD
out = model(x)
print(f"out (after SGD) = {out.data}")

# loss after SGD
loss = mae(out, targets)
print(f"loss (after SGD) = {loss.data}")