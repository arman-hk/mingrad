import numpy as np
from mingrad.engine import Value
from mingrad.nn import Linear, Sequential, ReLU, MAE, SGD

# data
x = Value(np.random.randn(4, 2))
y = Value(np.random.randn(4, 2))
print(f"x = {x.data}")
print(f"y = {y.data}")

# nn

model = Sequential(
    Linear(2, 4),
    ReLU(),
    Linear(4, 2)
)

optimizer = SGD(model.parameters(), lr=0.01)

mae = MAE()

for epoch in range(10):
    out = model(x)
    loss = mae(out, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.data}")
