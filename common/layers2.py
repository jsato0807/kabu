import numpy as np
from functools import reduce

class Variable:
    def __init__(self, value, requires_grad=True, parents=None):
        self.value = value
        self.requires_grad = requires_grad
        self.parents = parents or []  # list of (parent_var, local_grad_fn)
        self.grads = {}               # {wrt_variable: ∂wrt/∂self}
        self._backward_called = set()

    def backward(self, wrt=None, upstream_grad=1.0):
        if not self.requires_grad:
            return

        if wrt is None:
            wrt = self

        self.grads[wrt] = self.grads.get(wrt, 0.0) + upstream_grad

        if wrt in self._backward_called:
            return
        self._backward_called.add(wrt)

        for parent, local_grad_fn in self.parents:
            parent.backward(wrt=wrt, upstream_grad=local_grad_fn(upstream_grad))

    def grad(self, wrt):
        return self.grads.get(wrt, 0.0)



def sum_variables(vars):
    return reduce(add, vars)


def add(x, y):
    return Variable(x.value + y.value, parents=[
        (x, lambda g: g),
        (y, lambda g: g)
    ])

def sub(x, y):
    return Variable(x.value - y.value, parents=[
        (x, lambda g: g),
        (y, lambda g: -g)
    ])

def mul(x, y):
    return Variable(x.value * y.value, parents=[
        (x, lambda g: g * y.value),
        (y, lambda g: g * x.value)
    ])

def div(x, y):
    return Variable(x.value / y.value, parents=[
        (x, lambda g: g / y.value),
        (y, lambda g: -g * x.value / (y.value ** 2))
    ])



def exp(x):
    e = np.exp(x.value)
    return Variable(e, parents=[(x, lambda g: g * e)])

def log(x):
    return Variable(np.log(x.value), parents=[(x, lambda g: g / x.value)])

def sigmoid(x):
    s = 1 / (1 + np.exp(-x.value))
    return Variable(s, parents=[(x, lambda g: g * s * (1 - s))])

def tanh(x):
    t = np.tanh(x.value)
    return Variable(t, parents=[(x, lambda g: g * (1 - t ** 2))])

def relu(x):
    return Variable(x.value if x.value > 0 else 0.0,
                    parents=[(x, lambda g: g if x.value > 0 else 0.0)])

def softplus(x):
    sig = 1 / (1 + np.exp(-x.value))
    s = np.log(1 + np.exp(x.value))
    return Variable(s, parents=[(x, lambda g: g * sig)])


def min_var(x, y):
    if x.value < y.value:
        return Variable(x.value, parents=[
            (x, lambda g: g),
            (y, lambda g: 0.0)
        ])
    else:
        return Variable(y.value, parents=[
            (x, lambda g: 0.0),
            (y, lambda g: g)
        ])


def affine(x, w, b):
    """
    out = w * x + b
    ∂out/∂x = w
    ∂out/∂w = x
    ∂out/∂b = 1
    """
    return Variable(w.value * x.value + b.value, parents=[
        (x, lambda g: g * w.value),
        (w, lambda g: g * x.value),
        (b, lambda g: g)
    ])

if __name__  == "__main__":
    x = Variable(1.0)
    w = Variable(2.0)
    b = Variable(0.5)

    c = Variable(7)
    d = Variable(6)
    e = Variable(5)

    y = affine(x, w, b)      # y = 2.0 * 1.0 + 0.5 = 2.5

    t = add(mul(div(y,c),d),e)

    z = relu(t)

    z.backward()             # 自動的に wrt=z が設定される

    print("∂z/∂x =", x.grad(z))  # 2.0 if y > 0 else 0.0
    print("∂z/∂w =", w.grad(z))  # 1.0 if y > 0 else 0.0
    print("∂z/∂b =", b.grad(z))  # 1.0 if y > 0 else 0.0
    print("∂z/∂c =", c.grad(z))  
    print("∂z/∂d =", d.grad(z))  
    print("∂z/∂e =", e.grad(z))  