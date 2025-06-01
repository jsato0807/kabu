import numpy as np
from functools import reduce

grad_fns = {
    "add": lambda g, x, y: (g, g),
    "sub": lambda g, x, y: (g, -g),
    "mul": lambda g, x, y: (g * y.value, g * x.value),
    "div": lambda g, x, y: (g / y.value, -g * x.value / (y.value ** 2)),
    "exp": lambda g, x: (g * np.exp(x.value),),
    "log": lambda g, x: (g / x.value,),
    "sigmoid": lambda g, x: (
        s := 1 / (1 + np.exp(-x.value)),
        g * s * (1 - s),
    ),
    "tanh": lambda g, x: (
        t := np.tanh(x.value),
        g * (1 - t**2),
    ),
    "relu": lambda g, x: (g * (x.value > 0).astype(float),),
    "softplus": lambda g, x: (
        sig := 1 / (1 + np.exp(-x.value)),
        g * sig,
    ),
    "asinh": lambda g, x: (g / np.sqrt(x.value**2 + 1),),
    "sinh": lambda g, x: (g * np.cosh(x.value),),
    "min": lambda g, x, y: (g if x.value < y.value else 0.0, g if y.value < x.value else 0.0),
    "abs": lambda g, x: (g * np.sign(x.value),),
    "sign": lambda g, x: (np.zeros_like(g),),
    "affine": lambda g, x, w, b: (
        np.dot(g, w.value.T),
        np.outer(x.value, g),
        g
    ),
    "identity": lambda g, x: (g,)
}

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
            for parent, grad_fn_name in node.parents:
                if parent.requires_grad:
                    grad_fn = grad_fns[grad_fn_name]
                    local_grad = grad_fn(grad, node, parent)  # 必ず3引数渡す
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
    return Variable(x.value + y.value, parents=[(x, "add"), (y, "add")], name=f"add({x.name},{y.name})")

def sub(x, y):
    return Variable(x.value - y.value, parents=[(x, "sub"), (y, "sub")], name=f"sub({x.name},{y.name})")

def mul(x, y):
    return Variable(x.value * y.value, parents=[(x, "mul"), (y, "mul")], name=f"mul({x.name},{y.name})")

def div(x, y):
    return Variable(x.value / y.value, parents=[(x, "div"), (y, "div")], name=f"div({x.name},{y.name})")

def exp(x):
    return Variable(np.exp(x.value), parents=[(x, "exp")], name=f"exp({x.name})")

def log(x):
    return Variable(np.log(x.value), parents=[(x, "log")], name=f"log({x.name})")

def sigmoid(x):
    return Variable(1 / (1 + np.exp(-x.value)), parents=[(x, "sigmoid")], name=f"sigmoid({x.name})")

def tanh(x):
    return Variable(np.tanh(x.value), parents=[(x, "tanh")], name=f"tanh({x.name})")

def relu(x):
    return Variable(np.maximum(0, x.value), parents=[(x, "relu")], name=f"relu({x.name})")

def softplus(x):
    return Variable(np.log(1 + np.exp(x.value)), parents=[(x, "softplus")], name=f"softplus({x.name})")

def asinh(x):
    return Variable(np.arcsinh(x.value), parents=[(x, "asinh")], name=f"asinh({x.name})")

def sinh(x):
    return Variable(np.sinh(x.value), parents=[(x, "sinh")], name=f"sinh({x.name})")

def min_var(x, y):
    return Variable(min(x.value, y.value), parents=[(x, "min"), (y, "min")], name=f"min({x.name},{y.name})")

def abs_var(x):
    return Variable(np.abs(x.value), parents=[(x, "abs")], name=f"abs({x.name})")

def sign(x):
    return Variable(np.sign(x.value), parents=[(x, "sign")], name=f"sign({x.name})")

def affine(x, w, b):
    return Variable(np.dot(x.value, w.value) + b.value, parents=[(x, "affine"), (w, "affine"), (b, "affine")], name=f"affine({x.name},{w.name},{b.name})")

def stop_grad_with_identity(x):
    return Variable(x.value, parents=[(x, "identity")], name=f"identity({x.name})")


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