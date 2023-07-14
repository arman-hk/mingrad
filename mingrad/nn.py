import numpy as np
from engine import Value

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

""" Loss Functions """

class MAE:
    def __call__(self, pred, target):
        self.diff = pred - target
        return self.diff.abs().mean()

    def backward(self, grad=None):
        return self.diff.sign().data / self.diff.size

""" Optimization Algorithms """

class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def step(self):
        for p in self.params:
            if p is not None:
                p.data -= self.lr * p.grad
                p.grad = None # clear grad for the next iter

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
