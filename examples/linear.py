import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from mingrad.engine import Value
from mingrad.nn import Linear, Sequential, MAE, Adam
import matplotlib.pyplot as plt

# data
diabetes = load_diabetes()
x, y = diabetes.data, diabetes.target

# split + Value
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, y_train = Value(x_train), Value(y_train.reshape(-1, 1))


# model
model = Sequential(
    Linear(10, 1)
    )
optimizer = Adam(model.parameters(), lr=0.9)
mae = MAE()

# training loop
losses = []
for epoch in range(400):
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