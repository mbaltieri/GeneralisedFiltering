import torch

# Ozaki local linearisation scheme bridging continuous and discrete SDEs
# Given:
# t = time interval
# n = number of variables
# f = dx/dt
# J = Jac(f) = df/dx
# NB: Karl uses a different (and more efficient?) form that can be found in spm_dx
# NB: Using pinverse() instead of inverse() to deal with not full rank matrices
# Simple treatments can be found in the original papers by Ozaki, and the more recent
# 1) Iacus, Stefano M. Simulation and inference for stochastic differential equations: with R examples. Springer Science & Business Media, 2009.
# 2) Panik, Michael J. Stochastic Differential Equations: An Introduction with Applications in Population Dynamics Modeling. John Wiley & Sons, 2017.

def dx_ll(t, J, f):
    return (torch.matrix_exp(t * J) - torch.eye(len(J))) @ J.pinverse() @ f