import numpy as np
from functools import reduce

class Variable:
    _instances = set()  # 全インスタンスを保持

    def __init__(self, value, requires_grad=True, parents=None, name=None):
        self.value = np.array(value) if not isinstance(value, np.ndarray) else value
        self.requires_grad = requires_grad
        self.parents = parents or []  # list of (parent_var, local_grad_fn)
        self.grads = {}               # grads[wrt] = dL/dself
        self.name = name
        self.last_topo_order = None
        self.prev_grad = None

        Variable._instances.add(self)

    def build_topo_iterative(self):
        visited = set()
        topo_order = []
        stack = [(self, False)]

        while stack:
            var, processed = stack.pop()
            if var in visited:
                continue

            if processed:
                visited.add(var)
                topo_order.append(var)
            else:
                stack.append((var, True))  # 後で追加（処理対象）
                for parent, _ in var.parents:
                    stack.append((parent, False))

        return topo_order


    def backward(self, wrt=None, upstream_grad=None):
        if not self.requires_grad:
            return

        if wrt is None:
            wrt = self
        if upstream_grad is None:
            upstream_grad = np.ones_like(self.value)

        # 非再帰的トポロジカル順
        topo_order = self.build_topo_iterative()
        self.grads[wrt] = upstream_grad

        for node in reversed(topo_order):
            grad = node.grads.get(wrt, np.zeros_like(node.value))
            for parent, local_grad_fn in node.parents:
                if parent.requires_grad:
                    local_grad = local_grad_fn(grad)
                    prev_grad = parent.grads.get(wrt, np.zeros_like(parent.value))
                    parent.grads[wrt] = prev_grad + local_grad
        self.last_topo_order = topo_order
    
    def grad(self, wrt):
        g = self.grads.get(wrt, np.zeros_like(self.value))
        if np.isscalar(g):
            g = np.ones_like(self.value) * g
        return g
    
    @classmethod
    def clear_graph(cls):
        for v in list(cls._instances):
            v.parents = []
            v.grads = {}
        cls._instances.clear()

    def detach(self):
        return Variable(np.copy(self.value))
    
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)
    
    def __repr__(self):
        return f"Variable(name={self.name}, value={self.value})"
    
    def __del__(self):
        Variable._instances.discard(self)  # 削除時にもクリーンに


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
    ], name=f"add({x.name},{y.name})")

def sub(x, y):
    return Variable(x.value - y.value, parents=[
        (x, lambda g: g),
        (y, lambda g: -g)
    ], name=f"sub({x.name},{y.name})")

def mul(x, y):
    return Variable(x.value * y.value, parents=[
        (x, lambda g: g * y.value),
        (y, lambda g: g * x.value)
    ], name=f"mul({x.name},{y.name})")

def div(x, y):
    return Variable(x.value / y.value, parents=[
        (x, lambda g: g / y.value),
        (y, lambda g: -g * x.value / (y.value ** 2))
    ], name=f"div({x.name},{y.name})")



def exp(x):
    e = np.exp(x.value)
    return Variable(e, parents=[(x, lambda g: g * e)])

def log(x):
    return Variable(np.log(x.value), parents=[(x, lambda g: g / x.value)], name=f"exp({x.name})")

def sigmoid(x):
    s = 1 / (1 + np.exp(-x.value))
    return Variable(s, parents=[(x, lambda g: g * s * (1 - s))], name=f"sigmoid({x.name})")

def tanh(x):
    t = np.tanh(x.value)
    return Variable(t, parents=[(x, lambda g: g * (1 - t ** 2))], name=f"tanh({x.name})")

def relu(x):
    relu_val = np.maximum(0, x.value)
    def local_grad_fn(grad):
        return grad * (x.value > 0).astype(float)  # 要素ごとに勾配を 0 or 1 に

    return Variable(relu_val, parents=[(x, local_grad_fn)], name=f"relu{x.name}")

def softplus(x):
    sig = 1 / (1 + np.exp(-x.value))
    s = np.log(1 + np.exp(x.value))
    return Variable(s, parents=[(x, lambda g: g * sig)], name=f"softplus{x.name}")


def asinh(x):
    """
    y = arcsinh(x) = log(x + sqrt(x^2 + 1))
    dy/dx = 1 / sqrt(x^2 + 1)
    """
    s = np.arcsinh(x.value)
    return Variable(s, parents=[(x, lambda g: g / np.sqrt(x.value**2 + 1))], name=f"asinh{x.name}")

def sinh(x):
    """
    y = sinh(x) = (e^x - e^{-x}) / 2
    dy/dx = cosh(x) = (e^x + e^{-x}) / 2
    """
    s = np.sinh(x.value)
    return Variable(s, parents=[(x, lambda g: g * np.cosh(x.value))], name=f"sinh{x.name}")


def min_var(x, y):
    if x.value < y.value:
        return Variable(x.value, parents=[
            (x, lambda g: g),
            (y, lambda g: 0.0)
        ], name=f"min({x.name},{y.name})")
    else:
        return Variable(y.value, parents=[
            (x, lambda g: 0.0),
            (y, lambda g: g)
        ], name=f"min({x.name},{y.name})")
  
    
def abs_var(x):
    val = np.abs(x.value)

    def local_grad_fn(g):
        return g * np.sign(x.value)

    return Variable(val, parents=[(x, local_grad_fn)], name=f"abs({x.name})")



def sign(x):
    s = np.sign(x.value)
    return Variable(s, parents=[(x, lambda g: np.zeros_like(g))], name=f"sign({x.name})")


def affine(x, w, b):
    return Variable(np.dot(x.value, w.value) + b.value, parents=[
        (x, lambda g: np.dot(g, w.value.T)),
        (w, lambda g: np.outer(x.value, g)),
        (b, lambda g: g)
    ], name=f"affine({x.name},{w.name},{b.name})")

def stop_grad_with_identity(x):
    """
    Forward: 値そのものをそのまま返す
    Backward: 恒等関数（g → g）として勾配をそのまま返す（chain-ruleの伝播抑制）
    """
    return Variable(x.value, parents=[(x, lambda g: g * 1.0)])


if __name__  == "__main__":

    x = Variable(1.0)
    y = relu(stop_grad_with_identity(tanh(x)))  # tanh(x) の勾配は 1.0 として流す
    z = mul(y,Variable(2))

    z.backward()

    print(f"∂z/∂x ={x.grad(z)}")

    Variable.clear_graph()

    print(z.parents)
    print(x.grad(z))

    #x = Variable([1.0, 1.0], name="x")
    #w = Variable(2.0, name="w")
    #b = Variable(0.5, name="b")
    #c = Variable(7.0, name="c")
    #d = Variable(6.0, name="d")
    #e = Variable(5.0, name="e")
#
    #y = affine(x, w, b)      # y = 2.0 * 1.0 + 0.5 = 2.5
#
    #t = add(mul(div(y,c),d),e)
#
    #z = relu(t)
#
    #z.backward()             # 自動的に wrt=z が設定される
#
    #print("∂z/∂x =", x.grad(z))  # 2.0 if y > 0 else 0.0
    #print("∂z/∂w =", w.grad(z))  # 1.0 if y > 0 else 0.0
    #print("∂z/∂b =", b.grad(z))  # 1.0 if y > 0 else 0.0
    #print("∂z/∂c =", c.grad(z))  
    #print("∂z/∂d =", d.grad(z))  
    #print("∂z/∂e =", e.grad(z))  