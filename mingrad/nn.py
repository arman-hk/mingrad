import numpy as np
from mingrad.engine import Value

""" Linear Layer """

class Linear:
    def __init__(self, input_dim, output_dim):
        # init weights and biases
        self.weights = Value(np.random.randn(input_dim, output_dim) * 0.01)
        self.bias = Value(np.zeros(output_dim))

    def __call__(self, x):
        # forward pass
        self.x = x
        out = x @ self.weights + self.bias
        return out

    def backward(self, grad):
        # grads with respect to inputs and params
        self.weights.grad += self.x.T @ grad
        self.bias.grad += np.sum(grad, axis=0)
        grad_out = grad @ self.weights.T.data
        return grad_out

    def parameters(self):
        return [self.weights, self.bias]

""" Activation Functions """

class ReLU:
    def __call__(self, x):
        return x.relu()

class Tanh:
    def __call__(self, x):
        return x.tanh()

class Sigmoid:
    def __call__(self, x):
        return x.sigmoid()

""" Loss Functions """

class MAE:
    def __call__(self, pred, target):
        self.diff = pred - target
        return self.diff.abs().mean()

    def backward(self, grad=None):
        return self.diff.sign().data / self.diff.size

class MSE:
    def __call__(self, pred, target):
        self.diff = pred - target
        return (self.diff ** 2).mean()

    def backward(self, grad=None):
        grad = 1 if grad is None else grad
        return (2 * self.diff * grad).data / self.diff.data.size

class BCE:
    def __call__(self, pred, target):
        self.pred = pred = pred.data
        self.target = target = target.data
        loss = -(np.mean((target * np.log(pred)) + ((1 - target) * np.log(1 - pred))))
        return Value(loss)

    def backward(self, grad=None):
        grad = 1 if grad is None else grad
        grad_loss = (self.pred - self.target) / (self.pred * (1 - self.pred) * self.pred.shape[0]) * grad
        return grad_loss

""" Optimization Algorithms """

class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def step(self):
        for p in self.params:
            if p is not None:
                p.data -= self.lr * p.grad
                p.grad = np.zeros_like(p.data) # clear grad to zeroes

class Adam:
    # inspired by pytorch
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p is not None:
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p.grad
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (p.grad ** 2) # uncentered variance
                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                p.grad = np.zeros_like(p.data) # clear grad to zeroes

""" Container """

class Sequential:
    def __init__(self, *layers):
        # stores layers
        self.layers = layers

    def __call__(self, x):
        # forward pass on each layer with the output of the prev layer
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad):
        # backward pass on each layer but in reverse order
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
        return grad
    
    def parameters(self):
        # iterate over layers and collect their parameters
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params
