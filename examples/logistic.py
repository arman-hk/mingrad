import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from mingrad.engine import Value
from mingrad.nn import Linear, Sequential, MSE, SGD, Sigmoid
import matplotlib.pyplot as plt

# breast cancer dataset
data = load_breast_cancer()
x, y = data.data, data.target
y = y.reshape(-1, 1)

# split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Value
n_features = x_train.shape[1]
x_train, y_train, x_test = Value(x_train), Value(y_train), Value(x_test)

# logistic model
model = Sequential(
    Linear(n_features, 1),
    Sigmoid()
)

optimizer = SGD(model.parameters(), lr=0.9)
loss_func = MSE()

# training loop
losses = []
for epoch in range(100):
    out = model(x_train)
    loss = loss_func(out, y_train)
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

# pred + conf matrix
y_pred = model(x_test)
y_pred_binary = (y_pred.data > 0.5).astype(float)
cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()