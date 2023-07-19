import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mingrad.engine import Value
from mingrad.nn import Linear, Sequential, MAE, SGD
import matplotlib.pyplot as plt

# data
housing = fetch_california_housing()
x, y = housing.data, housing.target

# standard
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y.reshape(-1, 1))

# split + Value
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, y_train = Value(x_train), Value(y_train)

# model
model = Sequential(
    Linear(8, 1)
    )
optimizer = SGD(model.parameters(), lr=0.001)
mae = MAE()

# training loop
losses = []
for epoch in range(3):
    out = model(x_train)
    loss = mae(out, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.data}")
    losses.append(loss.data)

# loss curve
plt.plot(losses)
plt.title('Loss Curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()