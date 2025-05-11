import numpy as np
from functools import reduce

class Variable:
    def __init__(self, value, requires_grad=True):
        self.value = value
        self.requires_grad = requires_grad
        self.gradients = {}  # 任意の目的関数Lに対する ∂L/∂self を格納
        self._backward = lambda: None
        self._prev = set()

    def backward(self):
        self.gradients[self] = 1.0  # 自分自身に対する微分 ∂self/∂self = 1
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        for node in reversed(topo):
            node._backward()

    def grad(self, wrt):
        return self.gradients.get(wrt, 0.0)


def add(x, y):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad=False)
    if not isinstance(y, Variable):
        y = Variable(y, requires_grad=False)

    out = Variable(x.value + y.value)

    def _backward():
        for L, dL_dout in out.gradients.items():  # L は目的関数、dL_dout = ∂L/∂out
            if x.requires_grad:
                dL_dx = 1.0 * dL_dout  # ∂out/∂x = 1
                x.gradients[L] = x.gradients.get(L, 0.0) + dL_dx
            if y.requires_grad:
                dL_dy = 1.0 * dL_dout  # ∂out/∂y = 1
                y.gradients[L] = y.gradients.get(L, 0.0) + dL_dy

    out._prev = {x, y}
    out._backward = _backward
    return out


def sum_variables(vars):
    return reduce(add, vars)


def mul(x, y):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad=False)
    if not isinstance(y, Variable):
        y = Variable(y, requires_grad=False)

    out = Variable(x.value * y.value)

    def _backward():
        for L, dL_dout in out.gradients.items():
            if x.requires_grad:
                dL_dx = y.value * dL_dout
                x.gradients[L] = x.gradients.get(L, 0.0) + dL_dx
            if y.requires_grad:
                dL_dy = x.value * dL_dout
                y.gradients[L] = y.gradients.get(L, 0.0) + dL_dy

    out._prev = {x, y}
    out._backward = _backward
    return out


def sub(x, y):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad=False)
    if not isinstance(y, Variable):
        y = Variable(y, requires_grad=False)

    out = Variable(x.value - y.value)

    def _backward():
        for L, dL_dout in out.gradients.items():
            if x.requires_grad:
                dL_dx = 1.0 * dL_dout
                x.gradients[L] = x.gradients.get(L, 0.0) + dL_dx
            if y.requires_grad:
                dL_dy = -1.0 * dL_dout
                y.gradients[L] = y.gradients.get(L, 0.0) + dL_dy

    out._prev = {x, y}
    out._backward = _backward
    return out


def exp(x):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad=False)

    out = Variable(np.exp(x.value))

    def _backward():
        for L, dL_dout in out.gradients.items():
            if x.requires_grad:
                dL_dx = np.exp(x.value) * dL_dout
                x.gradients[L] = x.gradients.get(L, 0.0) + dL_dx

    out._prev = {x}
    out._backward = _backward
    return out


def div(x, y):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad=False)
    if not isinstance(y, Variable):
        y = Variable(y, requires_grad=False)

    out = Variable(x.value / y.value)

    def _backward():
        for L, dL_dout in out.gradients.items():
            if x.requires_grad:
                dL_dx = (1.0 / y.value) * dL_dout
                x.gradients[L] = x.gradients.get(L, 0.0) + dL_dx
            if y.requires_grad:
                dL_dy = (-x.value / (y.value ** 2)) * dL_dout
                y.gradients[L] = y.gradients.get(L, 0.0) + dL_dy

    out._prev = {x, y}
    out._backward = _backward
    return out


def log(x):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad=False)

    out = Variable(np.log(x.value))

    def _backward():
        for L, dL_dout in out.gradients.items():
            if x.requires_grad:
                dL_dx = (1.0 / x.value) * dL_dout
                x.gradients[L] = x.gradients.get(L, 0.0) + dL_dx

    out._prev = {x}
    out._backward = _backward
    return out


def minimum(a, b):
    out = a if a.value < b.value else b

    def _backward():
        if out is a:
            out.gradients[a] *= out.grad(a)
        else:
            out.gradients[b] *= out.grad(b)

    out._backward = _backward
    out._prev = {a, b}
    return out


def relu(x):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad=False)

    out = Variable(x.value if x.value > 0 else 0.0)

    def _backward():
        for L, dL_dout in out.gradients.items():
            if x.requires_grad:
                dL_dx = (1.0 if x.value > 0 else 0.0) * dL_dout
                x.gradients[L] = x.gradients.get(L, 0.0) + dL_dx

    out._prev = {x}
    out._backward = _backward
    return out

def tanh(x):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad=False)

    t = np.tanh(x.value)
    out = Variable(t)

    def _backward():
        for L, dL_dout in out.gradients.items():
            if x.requires_grad:
                dL_dx = (1 - t ** 2) * dL_dout
                x.gradients[L] = x.gradients.get(L, 0.0) + dL_dx

    out._prev = {x}
    out._backward = _backward
    return out


def sigmoid(x):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad=False)

    s = 1 / (1 + np.exp(-x.value))
    out = Variable(s)

    def _backward():
        for L, dL_dout in out.gradients.items():
            if x.requires_grad:
                dL_dx = s * (1 - s) * dL_dout
                x.gradients[L] = x.gradients.get(L, 0.0) + dL_dx

    out._prev = {x}
    out._backward = _backward
    return out


def softplus(x):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad=False)

    out_val = np.log(1 + np.exp(x.value))
    out = Variable(out_val)

    def _backward():
        s = 1 / (1 + np.exp(-x.value))  # sigmoid(x)
        for L, dL_dout in out.gradients.items():
            if x.requires_grad:
                dL_dx = s * dL_dout
                x.gradients[L] = x.gradients.get(L, 0.0) + dL_dx

    out._prev = {x}
    out._backward = _backward
    return out


def affine(x, w, b):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad=False)
    if not isinstance(w, Variable):
        w = Variable(w, requires_grad=False)
    if not isinstance(b, Variable):
        b = Variable(b, requires_grad=False)

    out = Variable(x.value * w.value + b.value)

    def _backward():
        for L, dL_dout in out.gradients.items():
            if x.requires_grad:
                dL_dx = w.value * dL_dout
                x.gradients[L] = x.gradients.get(L, 0.0) + dL_dx
            if w.requires_grad:
                dL_dw = x.value * dL_dout
                w.gradients[L] = w.gradients.get(L, 0.0) + dL_dw
            if b.requires_grad:
                dL_db = 1.0 * dL_dout
                b.gradients[L] = b.gradients.get(L, 0.0) + dL_db

    out._prev = {x, w, b}
    out._backward = _backward
    return out

