import numpy as np
from functools import reduce

class Variable:
    def __init__(self, value, requires_grad=True, parents=None):
        self.value = np.array(value) if not isinstance(value, np.ndarray) else value
        self.requires_grad = requires_grad
        self.parents = parents or []  # list of (parent_var, local_grad_fn)
        self.grads = {}               # grads[wrt] = dL/dself

    def backward(self, wrt=None, upstream_grad=None):
        if not self.requires_grad:
            return

        if wrt is None:
            wrt = self

        if upstream_grad is None:
            upstream_grad = np.ones_like(self.value)

        # === Step 1: トポロジカルソート ===
        topo_order = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent, _ in v.parents:
                    build_topo(parent)
                topo_order.append(v)

        build_topo(self)

        # === Step 2: 出発点に勾配を与える（∂L/∂L = 1）===
        init = self.grads.get(wrt, np.zeros_like(self.value))
        if np.isscalar(init):
            init = np.ones_like(self.value) * init
        self.grads[wrt] = init + upstream_grad

        # === Step 3: 逆順で逆伝播 ===
        for node in reversed(topo_order):
            grad = node.grads.get(wrt, np.zeros_like(node.value))
            if np.isscalar(grad):
                grad = np.ones_like(node.value) * grad

            for parent, local_grad_fn in node.parents:
                if parent.requires_grad:
                    local_grad = local_grad_fn(grad)
                    if np.isscalar(local_grad):
                        local_grad = np.ones_like(parent.value) * local_grad

                    prev = parent.grads.get(wrt, np.zeros_like(parent.value))
                    if np.isscalar(prev):
                        prev = np.ones_like(parent.value) * prev

                    parent.grads[wrt] = prev + local_grad

    def grad(self, wrt):
        g = self.grads.get(wrt, np.zeros_like(self.value))
        if np.isscalar(g):
            g = np.ones_like(self.value) * g
        return g
    
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for name in params:
            params[name].value -= self.lr * grads[name]

class RMSprop:
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.v = {}

    def update(self, params, grads):
        for name in params:
            if name not in self.v:
                self.v[name] = np.zeros_like(grads[name])
            self.v[name] = self.beta * self.v[name] + (1 - self.beta) * (grads[name] ** 2)
            params[name].value -= self.lr * grads[name] / (np.sqrt(self.v[name]) + self.epsilon)

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for name in params:
            if name not in self.m:
                self.m[name] = np.zeros_like(grads[name])
                self.v[name] = np.zeros_like(grads[name])
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grads[name] ** 2)

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            params[name].value -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)



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
    relu_val = np.maximum(0, x.value)
    def local_grad_fn(grad):
        return grad * (x.value > 0).astype(float)  # 要素ごとに勾配を 0 or 1 に

    return Variable(relu_val, parents=[(x, local_grad_fn)])

def softplus(x):
    sig = 1 / (1 + np.exp(-x.value))
    s = np.log(1 + np.exp(x.value))
    return Variable(s, parents=[(x, lambda g: g * sig)])


def asinh(x):
    """
    y = arcsinh(x) = log(x + sqrt(x^2 + 1))
    dy/dx = 1 / sqrt(x^2 + 1)
    """
    s = np.arcsinh(x.value)
    return Variable(s, parents=[(x, lambda g: g / np.sqrt(x.value**2 + 1))])

def sinh(x):
    """
    y = sinh(x) = (e^x - e^{-x}) / 2
    dy/dx = cosh(x) = (e^x + e^{-x}) / 2
    """
    s = np.sinh(x.value)
    return Variable(s, parents=[(x, lambda g: g * np.cosh(x.value))])


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
  
    
def abs_var(x):
    val = np.abs(x.value)

    def local_grad_fn(g):
        return g * np.sign(x.value)

    return Variable(val, parents=[(x, local_grad_fn)])



def sign(x):
    s = np.sign(x.value)
    return Variable(s, parents=[(x, lambda g: np.zeros_like(g))])


def affine(x, w, b):
    return Variable(np.dot(x.value, w.value) + b.value, parents=[
        (x, lambda g: np.dot(g, w.value.T)),
        (w, lambda g: np.outer(x.value, g)),
        (b, lambda g: g)
    ])

if __name__  == "__main__":
    x = Variable([1.0,1.0])
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