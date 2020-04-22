import autograd.numpy as np                 # autograd now implemented in JAX, but still missing functions used here
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

def smoothnessMatrix(emb, phi):
    embedding_orders = emb
    h = 0.0                                                       # lag
    derivative_order = (embedding_orders-1)*2+1
    rho_tilde = np.zeros(derivative_order,)                     # array of autocorrelations
    S_inv = np.zeros((embedding_orders,embedding_orders))             # temporal roughness
    S = np.zeros((embedding_orders,embedding_orders))             # temporal smoothness

    drho = findDerivatives(derivative_order)
    rho_tilde[0] = rho(h, phi)
    for i in range(1, derivative_order):
        # rho_tilde[i] = dnrho(h, phi, i-1)
        rho_tilde[i] = drho[i-1](h, phi)

    for i in range(embedding_orders):
        for j in range(embedding_orders):
            S_inv[i, j] = np.power(-1, (i*j))*rho_tilde[i+j]
    S = np.linalg.inv(S_inv)

    return S

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