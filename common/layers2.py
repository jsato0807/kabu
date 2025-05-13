import numpy as np
from functools import reduce

class Variable:
    def __init__(self, value, requires_grad=True, parents=None, grad_fn=None):
        self.value = value
        self.requires_grad = requires_grad
        self.parents = parents or []  # [(parent, local_grad_fn)]
        self.grad_fn = grad_fn
        self.grads = {}  # dict: {wrt_variable: ∂wrt/∂self}
        self._backward_called = set()  # to avoid duplicate traversal

    def backward(self, wrt=None, upstream_grad=1.0):
        if not self.requires_grad:
            return

        if wrt is None:
            wrt = self

        self.grads[wrt] = self.grads.get(wrt, 0.0) + upstream_grad

        if wrt in self._backward_called:
            return
        self._backward_called.add(wrt)

        if self.grad_fn:
            for parent, local_grad in self.grad_fn(upstream_grad):
                parent.backward(wrt=wrt, upstream_grad=local_grad)

    def grad(self, wrt):
        return self.grads.get(wrt, 0.0)


def sum_variables(vars):
    return reduce(add, vars)


def add(x, y):
    out = Variable(x.value + y.value, parents=[(x, lambda g: g), (y, lambda g: g)])
    out.grad_fn = lambda g: [(x, g), (y, g)]
    return out

def sub(x, y):
    out = Variable(x.value - y.value, parents=[(x, lambda g: g), (y, lambda g: -g)])
    out.grad_fn = lambda g: [(x, g), (y, -g)]
    return out

def mul(x, y):
    out = Variable(x.value * y.value, parents=[(x, lambda g: g * y.value), (y, lambda g: g * x.value)])
    out.grad_fn = lambda g: [(x, g * y.value), (y, g * x.value)]
    return out

def div(x, y):
    out = Variable(x.value / y.value, parents=[(x, lambda g: g / y.value),
                                               (y, lambda g: -g * x.value / (y.value ** 2))])
    out.grad_fn = lambda g: [(x, g / y.value), (y, -g * x.value / (y.value ** 2))]
    return out


def exp(x):
    e = np.exp(x.value)
    out = Variable(e, parents=[(x, lambda g: g * e)])
    out.grad_fn = lambda g: [(x, g * e)]
    return out

def log(x):
    out = Variable(np.log(x.value), parents=[(x, lambda g: g / x.value)])
    out.grad_fn = lambda g: [(x, g / x.value)]
    return out

def sigmoid(x):
    s = 1 / (1 + np.exp(-x.value))
    out = Variable(s, parents=[(x, lambda g: g * s * (1 - s))])
    out.grad_fn = lambda g: [(x, g * s * (1 - s))]
    return out

def tanh(x):
    t = np.tanh(x.value)
    out = Variable(t, parents=[(x, lambda g: g * (1 - t ** 2))])
    out.grad_fn = lambda g: [(x, g * (1 - t ** 2))]
    return out

def relu(x):
    out_val = x.value if x.value > 0 else 0.0
    out = Variable(out_val, parents=[(x, lambda g: g if x.value > 0 else 0.0)])
    out.grad_fn = lambda g: [(x, g if x.value > 0 else 0.0)]
    return out

def softplus(x):
    s = np.log(1 + np.exp(x.value))
    sig = 1 / (1 + np.exp(-x.value))
    out = Variable(s, parents=[(x, lambda g: g * sig)])
    out.grad_fn = lambda g: [(x, g * sig)]
    return out


def min_var(x, y):
    if x.value < y.value:
        out = Variable(x.value, parents=[(x, lambda g: g), (y, lambda g: 0.0)])
        out.grad_fn = lambda g: [(x, g), (y, 0.0)]
    else:
        out = Variable(y.value, parents=[(x, lambda g: 0.0), (y, lambda g: g)])
        out.grad_fn = lambda g: [(x, 0.0), (y, g)]
    return out


def affine(x, w, b):
    out_val = w.value * x.value + b.value
    out = Variable(out_val, parents=[
        (x, lambda g: g * w.value),
        (w, lambda g: g * x.value),
        (b, lambda g: g)
    ])
    out.grad_fn = lambda g: [
        (x, g * w.value),
        (w, g * x.value),
        (b, g)
    ]
    return out


