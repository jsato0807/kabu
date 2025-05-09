import math
from functools import reduce

class Variable:
    def __init__(self, value, requires_grad=True):
        self.value = value
        self.requires_grad = requires_grad
        self.gradients = {}  # 任意の変数に関する微分 ∂self/∂wrt を保持
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


def add(a, b):
    out = Variable(a.value + b.value)
    out._prev = {a, b}
    def _backward():
        if a.requires_grad:
            a.gradients[wrt := a] = a.grad(wrt) + out.grad(out)
        if b.requires_grad:
            b.gradients[wrt := b] = b.grad(wrt) + out.grad(out)
    out._backward = _backward
    return out

def sum_variables(vars):
    return reduce(add, vars)

def mul(a, b):
    out = Variable(a.value * b.value)
    out._prev = {a, b}
    def _backward():
        if a.requires_grad:
            a.gradients[wrt := a] = a.grad(wrt) + b.value * out.grad(out)
        if b.requires_grad:
            b.gradients[wrt := b] = b.grad(wrt) + a.value * out.grad(out)
    out._backward = _backward
    return out

def sub(a, b):
    out = Variable(a.value - b.value)
    out._prev = {a, b}
    def _backward():
        if a.requires_grad:
            a.gradients[a] = a.grad(a) + 1.0 * out.grad(out)
        if b.requires_grad:
            b.gradients[b] = b.grad(b) - 1.0 * out.grad(out)
    out._backward = _backward
    return out

def div(a, b):
    out = Variable(a.value / b.value)
    out._prev = {a, b}
    def _backward():
        if a.requires_grad:
            a.gradients[a] = a.grad(a) + (1.0 / b.value) * out.grad(out)
        if b.requires_grad:
            b.gradients[b] = b.grad(b) - (a.value / (b.value ** 2)) * out.grad(out)
    out._backward = _backward
    return out





def relu(x):
    out = Variable(max(0, x.value))
    out._prev = {x}
    def _backward():
        dx = 1.0 if x.value > 0 else 0.0
        x.gradients[x] = x.grad(x) + dx * out.grad(out)
    out._backward = _backward
    return out

def tanh(x):
    t = math.tanh(x.value)
    out = Variable(t)
    out._prev = {x}
    def _backward():
        dx = 1 - t ** 2
        x.gradients[x] = x.grad(x) + dx * out.grad(out)
    out._backward = _backward
    return out

def sigmoid(x):
    s = 1 / (1 + math.exp(-x.value))
    out = Variable(s)
    out._prev = {x}
    def _backward():
        dx = s * (1 - s)
        x.gradients[x] = x.grad(x) + dx * out.grad(out)
    out._backward = _backward
    return out

def softplus(x):
    out = Variable(math.log(1 + math.exp(x.value)))
    out._prev = {x}
    def _backward():
        dx = 1 / (1 + math.exp(-x.value))
        x.gradients[x] = x.grad(x) + dx * out.grad(out)
    out._backward = _backward
    return out

def logscale(x, scale):
    out = Variable(x.value * scale)
    out._prev = {x}
    def _backward():
        dx = scale
        x.gradients[x] = x.grad(x) + dx * out.grad(out)
    out._backward = _backward
    return out

def affine(x, w, b):
    out = Variable(x.value * w.value + b.value)
    out._prev = {x, w, b}
    def _backward():
        x.gradients[x] = x.grad(x) + w.value * out.grad(out)
        w.gradients[w] = w.grad(w) + x.value * out.grad(out)
        b.gradients[b] = b.grad(b) + 1.0 * out.grad(out)
    out._backward = _backward
    return out
