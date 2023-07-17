import numpy as np
from mingrad.engine import Value
from mingrad.nn import Linear, Sequential, ReLU, MAE, SGD
import matplotlib.pyplot as plt

# data
a = [[-1.46626439, -0.14547705],
[-0.09220323, -0.04453503],
[-0.56252296, -0.49160473],
[0.76057888, -1.33456016]]

b = [[1.51414334, 0.41878582],
[1.06901496, -1.10981324],
[-1.79410769, 2.07008861],
[0.68137226,  1.35805047]]

x = Value(np.array(a))
y = Value(np.array(b))

# nn
model = Sequential(
    Linear(2, 2)
)

optimizer = SGD(model.parameters(), lr=0.3)

mae = MAE()

# training loop
losses = []
for epoch in range(10):
    out = model(x)
    loss = mae(out, y)
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