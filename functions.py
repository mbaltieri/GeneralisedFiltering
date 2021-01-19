# This module will contain general functions used for the inversion of generative models

import math

import torch
# import autograd.numpy as np                 # autograd now implemented in JAX, can be replaced
# from autograd import grad                   # move this to pure pytorch
from scipy.linalg import sqrtm, cholesky, toeplitz

torch.set_default_dtype(torch.float64)

### FUNCTIONS ###

def symsqrt(matrix):                # code from https://github.com/pytorch/pytorch/issues/25481, while we still wait for sqrtm
                                    # SVD > Cholemsky since the obtained "square root" is symmetric, and not lower or upper triangular
    """Compute the square root of a positive definite matrix."""
    _, s, v = matrix.svd()
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


def kronecker(A, B):                # code from https://discuss.pytorch.org/t/kronecker-product/3919/9, torch.kron is almost ready (probably torch 1.8?)
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))


def Diff(x, dimension, embeddings, shift=1):
    # D = kronecker(torch.eye(embeddings), torch.from_numpy(np.eye(dimension, k = shift)))    # TODO: torch does not support arbitrary diagonal shifts for the 'eye' function, numpy does
    offdiag = torch.diag(torch.ones(dimension), diagonal=shift)
    if shift >= 0:
        D = kronecker(torch.eye(embeddings), offdiag[:-shift,:-shift])    # TODO: find better workaround
    else:
        D = kronecker(torch.eye(embeddings), offdiag[:shift,:shift])    # TODO: find better workaround
    return D @ x




def f(x, u, a, A, B_u, B_a):
    # TODO: generalise this to include nonlinear treatments
    try:
        return A @ x + B_u @ u + B_a @ a
    except RuntimeError:
        print("Dimensions don't match!")
        return