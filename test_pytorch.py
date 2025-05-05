import torch
from torch import nn

class GWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h_fn, g_fn):
        with torch.no_grad():
            h_out = h_fn(x)
        ctx.save_for_backward(x)
        ctx.h_fn = h_fn
        ctx.g_fn = g_fn
        return g_fn(h_out.detach())

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x = x.detach().requires_grad_(True)
        h = ctx.h_fn(x)
        y = ctx.g_fn(h)
        y.backward(grad_output)
        return x.grad, None, None  # h_fn, g_fn は勾配不要

# 使用例
def h_fn(x):
    return torch.sin(x)

def g_fn(h_out):
    return h_out ** 2

x = torch.tensor([1.0], requires_grad=True)
y = GWrapper.apply(x, h_fn, g_fn)
y.backward()
print(f"x.grad: {x.grad}")
