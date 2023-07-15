import numpy as np

class Context:
    """Stores given arrays for a future call to Value.backward"""
    
    def __init__(self):
        self.saved_arrays = {}

    def save_for_backward(self, *keys):
        for idx, key in enumerate(keys):
            self.saved_arrays[idx] = key

class Value:
    """A node in the computation graph"""
    
    def __init__(self, data, _children=(), _grad_fn=None):
        self.data = np.array(data)
        self.grad = np.zeros_like(data, dtype=float)
        # internal variables
        self._prev = set(_children)
        self._grad_fn = _grad_fn
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ctx = Context()
        ctx.save_for_backward(self.data, other.data)

        def _grad_fn(grad):
            self_data, other_data = ctx.saved_arrays.values()
            self.grad += grad
            other.grad += grad
        out = Value(self.data + other.data, (self, other), _grad_fn)
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ctx = Context()
        ctx.save_for_backward(self.data, other.data)

        def _grad_fn(grad):
            self_data, other_data = ctx.saved_arrays.values()
            self.grad += other_data * grad
            other.grad += self_data * grad
        out = Value(self.data * other.data, (self, other), _grad_fn)
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ctx = Context()
        ctx.save_for_backward(self.data, other.data)

        def _grad_fn(grad):
            self_data, other_data = ctx.saved_arrays.values()
            self.grad += (other_data * self_data ** (other_data - 1)) * grad
            if np.any(self_data > 0):
                other.grad += np.sum((self.data ** other_data * np.log(self_data)) * grad)
            else:
                other.grad += 0
        out = Value(self.data ** other.data, (self, other), _grad_fn)
        return out

    def __neg__(self):
        def _grad_fn(grad):
            self.grad -= grad
        out = Value(-self.data, (self,), _grad_fn)
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        def _grad_fn(grad):
            self.grad += grad
            other.grad -= grad
        out = Value(self.data - other.data, (self, other), _grad_fn)
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ctx = Context()
        ctx.save_for_backward(self.data, other.data)

        def _grad_fn(grad):
            self_data, other_data = ctx.saved_arrays.values()
            self.grad += grad / other_data
            other.grad += -self_data / (other_data ** 2) * grad
        out = Value(self.data / other.data, (self, other), _grad_fn)
        return out

    def sqrt(self):
        def _grad_fn(grad):
            self.grad += grad / (2 * np.sqrt(self.data))
        out = Value(np.sqrt(self.data), (self,), _grad_fn)
        return out

    def exp(self):
        def _grad_fn(grad):
            self.grad += np.exp(self.data) * grad
        out = Value(np.exp(self.data), (self,), _grad_fn)
        return out

    def log(self):
        def _grad_fn(grad):
            self.grad += grad / self.data
        out = Value(np.log(self.data), (self,), _grad_fn)
        return out

    def abs(self):
        def _grad_fn(grad):
            self.grad += np.sign(self.data) * grad
        out = Value(np.abs(self.data), (self,), _grad_fn)
        return out

    def __matmul__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        
        def _grad_fn(grad):
            self.grad += grad @ other.data.T
            other.grad += self.data.T @ grad

        out = Value(self.data @ other.data, (self, other), _grad_fn)
        return out

    def __rmatmul__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        
        return other @ self

    @property
    def T(self):
        return Value(np.transpose(self.data))

    def relu(self):
        def _grad_fn(grad):
            self.grad += (self.data > 0) * grad
        out = Value(np.maximum(0, self.data), (self,), _grad_fn)
        return out

    def tanh(self):
        def _grad_fn(grad):
            self.grad += (1 - (np.tanh(self.data))**2) * grad
        out = Value(np.tanh(self.data), (self, ), _grad_fn)
        return out

    def sigmoid(self):
        def S(x):
            return 1/(1 + np.exp(-x))
        
        def _grad_fn(grad):
            self.grad += (S(self.data) * (1 - S(self.data))) * grad
        out = Value(S(self.data), (self,), _grad_fn)
        return out
    
    def mean(self):
        def _grad_fn(grad):
            self.grad += grad / np.size(self.data)
        out = Value(np.mean(self.data), (self,), _grad_fn)
        return out

    def backward(self, grad=None):
        if grad is None: grad = np.ones_like(self.data, dtype=float)
        self.grad = grad
        if self._grad_fn is not None: self._grad_fn(grad)
        # topo
        for child in self._prev:
            child.backward(child.grad)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
