# common/grad_cache.py
_grad_fn_cache = {}

def get_identity_grad_fn():
    if "identity" not in _grad_fn_cache:
        _grad_fn_cache["identity"] = lambda g: g
    return _grad_fn_cache["identity"]

def get_neg_identity_grad_fn():
    if "neg_identity" not in _grad_fn_cache:
        _grad_fn_cache["neg_identity"] = lambda g: -g
    return _grad_fn_cache["neg_identity"]

def get_index_grad_fn(i):
    key = f"index_{i}"
    if key not in _grad_fn_cache:
        _grad_fn_cache[key] = lambda g, idx=i: g[idx]
    return _grad_fn_cache[key]

def get_scalar_mul_grad_fn(c):
    key = f"mul_{c}"
    if key not in _grad_fn_cache:
        _grad_fn_cache[key] = lambda g, const=c: const * g
    return _grad_fn_cache[key]

def get_div_grad_fn(denom):
    key = f"div_{denom}"
    if key not in _grad_fn_cache:
        _grad_fn_cache[key] = lambda g, d=denom: g / d
    return _grad_fn_cache[key]

def get_sub_grad_fn(x_or_y, is_negative):
    return get_neg_identity_grad_fn() if is_negative else get_identity_grad_fn()

def get_zero_grad_fn():
    if "zero" not in _grad_fn_cache:
        _grad_fn_cache["zero"] = lambda g: 0.0
    return _grad_fn_cache["zero"]

def get_sign_grad_fn(val):
    key = f"sign_{id(val)}"
    if key not in _grad_fn_cache:
        _grad_fn_cache[key] = lambda g, v=val: g * np.sign(v)
    return _grad_fn_cache[key]
