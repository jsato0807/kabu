import torch

class GWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h_fn, g_fn):
        with torch.no_grad():
            h_out = h_fn(x)  # h(x): グラフから切り離す
        h_out_detached = h_out.detach().requires_grad_()

        with torch.enable_grad():
            y = g_fn(h_out_detached)  # g(h): グラフ付き計算

        ctx.save_for_backward(h_out_detached, y)
        ctx.g_fn = g_fn
        return y  # y には grad_fn が付き、.backward() 可能

    @staticmethod
    def backward(ctx, grad_output):
        h_out_detached, y = ctx.saved_tensors
        g_fn = ctx.g_fn

        grad_h = torch.autograd.grad(
            y, h_out_detached, grad_output, retain_graph=True, allow_unused=True
        )[0]

        # ∂y/∂x = 0 と明示
        return None, None, grad_h  # x, h_fn の勾配は不要

# 使用例
def h_fn(x):
    return torch.sin(x)

def g_fn(h):
    return h ** 2

x = torch.tensor([1.0], requires_grad=True)
y = GWrapper.apply(x, h_fn, g_fn)
y.backward()    # grad_output = 1.0 が自動で使われる（スカラー出力なら）

print(f"x.grad: {x.grad}")  # None になる（h'(x)を切ってるので）
