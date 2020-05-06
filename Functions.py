# This module will contain general functions used for the inversion of generative models

import torch
import autograd.numpy as np                 # autograd now implemented in JAX, but still missing functions used here (December 2019)
import matplotlib.pyplot as plt
import scipy.linalg as splin
from autograd import grad

### FUNCTIONS ###

def sigmoid(x):
#    return x
    return np.tanh(x)

def dsigmoid(x):
#    return x
    return 1 - np.tanh(x)**2

# autocorrelation function (Gaussian)
def rho(h, phi):
    return np.exp(-.5*h**2/phi)

# n-th derivative of rho function
def findDerivatives(orders):                   
    drho = [grad(rho,0)]
    for i in range(1,orders):
        drho.append(grad(drho[i-1]))

    return drho

def dnrho(h, phi, degree):
    return drho[degree](h, phi)

# TODO: This function is extremely slow at the moment due to the recurrent automatic 
# differentiation of the autocorrelation function (anything around 10 takes several second)
def smoothnessMatrix(embedding_orders, phi):
    h = 0.0                                                             # lag
    derivative_order = (embedding_orders-1)*2+1
    rho_tilde = np.zeros(derivative_order,)                             # array of autocorrelations
    S_inv = np.zeros((embedding_orders,embedding_orders))               # temporal roughness
    S = np.zeros((embedding_orders,embedding_orders))                   # temporal smoothness

    drho = findDerivatives(derivative_order)
    rho_tilde[0] = rho(h, phi)
    for i in range(1, derivative_order):
        # rho_tilde[i] = dnrho(h, phi, i-1)
        rho_tilde[i] = drho[i-1](h, phi)

    for i in range(embedding_orders):
        for j in range(embedding_orders):
            S_inv[i, j] = np.power(-1, (i*j))*rho_tilde[i+j]
    S = np.linalg.inv(S_inv)

    return torch.from_numpy(S)

## free energy functions ##
# generative process
def g(x, v):
    return F_tilde @ x + G_tilde @ v

def f(x, v):
    return np.dot(A_tilde, x) + np.dot(B_tilde, v)
#    return np.dot(A, x) + np.dot(B, sigmoid(a)) + np.dot(B, v)

# generative model
def g_gm(x, v, F_gm):
    return np.dot(F_gm, x) + np.dot(G_gm, v)

def f_gm(x, v, A_gm, B_gm):
    # no action in generative model, a = 0.0
    return np.dot(A_gm, x) + np.dot(B_gm, v)

def getObservation(x, v, w):
    x[:, 1:] = f(x[:, :-1], v) + np.dot(C, w[:, None])
    x[:, 0] += dt * x[:, 1]
    return g(x, v)
                

def Diff(x, variables_n, embeddings_n):
    D = np.eye(variables_n, k=1)
    I = np.eye(embeddings_n)
    D = np.kron(I, D)
    res = D @ x
    return res


def FreeEnergy(y, mu_x, mu_v, mu_pi_z, mu_pi_w, A_gm, B_gm, F_gm):
    return .5 * (np.sum(np.dot(np.dot((y - np.dot(F_gm, mu_x)).transpose(), mu_pi_z), (y - np.dot(F_gm, mu_x)))) + \
                np.sum(np.dot(np.dot((mu_x[:, 1:] - f_gm(mu_x[:, :-1], mu_v[:, :-1], A_gm, B_gm)).transpose(), mu_pi_w), (mu_x[:, 1:] - f_gm(mu_x[:, :-1], mu_v[:, :-1], A_gm, B_gm)))) - \
                np.trace(np.log(mu_pi_z * mu_pi_w)))

def F(y, x, v, theta, Pi_z, Pi_w, Pi_v, p_Pi_z, p_Pi_w):
    return .5 * (torch.bmm(torch.bmm(torch.transpose(y - g(x, v, theta), 1, 2), Pi_z), y - g(x, theta, v)) + \
                 torch.bmm(torch.bmm(torch.transpose(D(x, e_n) - f(x, v, theta), 1, 2), Pi_w), x[:,:,1:] - f(x, theta, v)) + \
                 torch.bmm(torch.bmm(torch.transpose(v - h(x, theta, v), 1, 2), Pi_v), v - h(x, theta, v)) + \
                 torch.bmm(torch.bmm(torch.transpose(Pi_z - p_1(x, theta, v), 1, 2), p_Pi_z), Pi_z - p_1(x, theta, v)) + \
                 torch.bmm(torch.bmm(torch.transpose(Pi_w - p_2(x, theta, v), 1, 2), p_Pi_w), Pi_w - p_2(x, theta, v)) + \
                 torch.log(torch.bmm(torch.bmm(torch.potrf(Pi_z).diag().prod(), torch.potrf(Pi_w).diag().prod()), torch.potrf(Pi_v).diag().prod())))