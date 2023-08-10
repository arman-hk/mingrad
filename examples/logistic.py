import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from mingrad.engine import Value
from mingrad.nn import Linear, Sequential, BCE, SGD, Sigmoid
import matplotlib.pyplot as plt

# breast cancer dataset
data = load_breast_cancer()
x, y = data.data, data.target
y = y.reshape(-1, 1)

# split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Value
n_features = x_train.shape[1]
x_train, y_train = Value(x_train), Value(y_train.astype(float))

# logistic model
model = Sequential(
    Linear(n_features, 1),
    Sigmoid()
)

optimizer = SGD(model.parameters(), lr=0.1)
loss_func = BCE()

# training loop
losses = []
for epoch in range(4):
    out = model(x_train)
    loss = loss_func(out, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.data}")
    losses.append(loss.data)

plt.plot(losses)
plt.title('Loss Curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
