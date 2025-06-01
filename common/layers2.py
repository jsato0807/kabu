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
                    local_grad = local_grad_fn(grad, node, parent)  # 必ず3引数渡す
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

    def __getitem__(self, idx):
        val = self.value[idx]

        def grad_fn(g, node, parent):
            # スカラー勾配 g を、元の shape に合わせて one-hot 勾配に変換
            full_grad = np.zeros_like(parent.value)
            full_grad[idx] = g
            return full_grad

        return Variable(val, parents=[(self, grad_fn)], name=f"{self.name}[{idx}]")



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
    def grad_fn_x(g, node, parent): return g
    def grad_fn_y(g, node, parent): return g
    return Variable(x.value + y.value, parents=[(x, grad_fn_x), (y, grad_fn_y)], name=f"add({x.name},{y.name})")

def sub(x, y):
    def grad_fn_x(g, node, parent): return g
    def grad_fn_y(g, node, parent): return -g
    return Variable(x.value - y.value, parents=[(x, grad_fn_x), (y, grad_fn_y)], name=f"sub({x.name},{y.name})")

def mul(x, y):
    def grad_fn_x(g, node, parent): return g * y.value
    def grad_fn_y(g, node, parent): return g * x.value
    return Variable(x.value * y.value, parents=[(x, grad_fn_x), (y, grad_fn_y)], name=f"mul({x.name},{y.name})")

def div(x, y):
    def grad_fn_x(g, node, parent): return g / y.value
    def grad_fn_y(g, node, parent): return -g * x.value / (y.value ** 2)
    return Variable(x.value / y.value, parents=[(x, grad_fn_x), (y, grad_fn_y)], name=f"div({x.name},{y.name})")

def relu(x):
    def grad_fn(g, node, parent): return g * (x.value > 0).astype(float)
    return Variable(np.maximum(0, x.value), parents=[(x, grad_fn)], name=f"relu({x.name})")

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x.value))
    def grad_fn(g, node, parent): return g * sig * (1 - sig)
    return Variable(sig, parents=[(x, grad_fn)], name=f"sigmoid({x.name})")

def tanh(x):
    th = np.tanh(x.value)
    def grad_fn(g, node, parent): return g * (1 - th ** 2)
    return Variable(th, parents=[(x, grad_fn)], name=f"tanh({x.name})")

def exp(x):
    ex = np.exp(x.value)
    def grad_fn(g, node, parent): return g * ex
    return Variable(ex, parents=[(x, grad_fn)], name=f"exp({x.name})")

def log(x):
    def grad_fn(g, node, parent): return g / x.value
    return Variable(np.log(x.value), parents=[(x, grad_fn)], name=f"log({x.name})")

def softplus(x):
    val = np.log1p(np.exp(x.value))
    def grad_fn(g, node, parent): return g * (1 / (1 + np.exp(-x.value)))
    return Variable(val, parents=[(x, grad_fn)], name=f"softplus({x.name})")

def asinh(x):
    val = np.arcsinh(x.value)
    def grad_fn(g, node, parent): return g / np.sqrt(x.value ** 2 + 1)
    return Variable(val, parents=[(x, grad_fn)], name=f"asinh({x.name})")

def sinh(x):
    val = np.sinh(x.value)
    def grad_fn(g, node, parent): return g * np.cosh(x.value)
    return Variable(val, parents=[(x, grad_fn)], name=f"sinh({x.name})")

def min_var(x, y):
    val = np.minimum(x.value, y.value)

    def grad_fn_x(g, node, parent):
        mask = x.value < y.value
        return g * mask.astype(float)

    def grad_fn_y(g, node, parent):
        mask = x.value >= y.value
        return g * mask.astype(float)

    return Variable(val, parents=[(x, grad_fn_x), (y, grad_fn_y)], name=f"min({x.name},{y.name})")

def abs_var(x):
    val = np.abs(x.value)
    def grad_fn(g, node, parent): return g * np.sign(x.value)
    return Variable(val, parents=[(x, grad_fn)], name=f"abs({x.name})")

def sign(x):
    def grad_fn(g, node, parent): return np.zeros_like(x.value)
    return Variable(np.sign(x.value), parents=[(x, grad_fn)], name=f"sign({x.name})")

def identity(x):
    def grad_fn(g, node, parent): return g
    return Variable(x.value, parents=[(x, grad_fn)], name=f"identity({x.name})")


def affine(x, w, b):
    def grad_fn_x(g, node, parent): return g * w.value
    def grad_fn_w(g, node, parent): return g * x.value
    def grad_fn_b(g, node, parent): return g
    return Variable(x.value * w.value + b.value,
                    parents=[(x, grad_fn_x), (w, grad_fn_w), (b, grad_fn_b)],
                    name=f"affine({x.name},{w.name},{b.name})")



if __name__  == "__main__":

    x = Variable(1.0)
    y = relu(identity(tanh(x)))  # tanh(x) の勾配は 1.0 として流す
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