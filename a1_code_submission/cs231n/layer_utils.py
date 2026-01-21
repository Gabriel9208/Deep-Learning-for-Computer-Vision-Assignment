from .layers import *
import numpy as np

def affine_forward(x, w, b):
  _x = x.reshape(x.shape[0], -1)
  _b = b.reshape(1, -1)
  z = _x @ w
  out = z + _b # (2, 3)
  cache = (x, w, z, out)
  return out, cache

def affine_backward(dout, fc_cache):
  x, w, z, out = fc_cache
  dx = dout @ w.T # (2, 3) (3, 4 * 5 * 6)
  dx = dx.reshape(*x.shape)
  dw = x.reshape(x.shape[0], -1).T @ dout # (4 * 5 * 6, 2) (2, 3)
  db = dout.sum(axis=0)
  return dx, dw, db

def relu_forward(x):
  out = np.maximum(0, x)
  cache = x
  return out, cache

def relu_backward(dout, fc_cache):
  x = fc_cache
  dx = dout * (x > 0)
  return dx

def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

