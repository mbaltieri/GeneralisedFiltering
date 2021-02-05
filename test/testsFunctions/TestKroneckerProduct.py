import autograd.numpy as np

np.random.seed(42)

a_size = 3
b_size = 4

a = np.random.randn(a_size,a_size)
a = np.eye(a_size)
b = np.array([[10, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
c = np.kron(a, b)
print(c)

block_row = 0
block_column = 2

print(c[block_row*a_size:(block_row+1)*a_size, block_column*b_size:(block_column+1)*b_size])
# print(c[])

hidden_states = 2

dt = 0.01
T = 10.
iterations = int(T/dt)

delta = .150                                                # parameters simulating a simple mass-spring system as the environment
epsilon = 15.0

gamma_z = 10
gamma_w = 4

sigma_z = np.exp(-gamma_z)
sigma_w = np.exp(-gamma_w)

A = np.array([[0, 1], [- delta, - epsilon]])                   # state transition matrix
B = np.array([[0, 0, 0], [1, 0, 0]])                        # input matrix
C = sigma_w * np.array([[0, 0], [0, 1]])               # noise dynamics matrix
D = sigma_z * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])             # noise observations matrix
F = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])               # measurement matrix
G = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0], [0, 0, 1]])

obs_states = 4
hidden_states = 2                                           # x, in Friston's work
hidden_causes = 3                                           # v, in Friston's work
embed_orders_obs = 2
embed_orders_states = 2                                      # generalised coordinates for hidden states x, but only using n-1
embed_orders_causes = 1                                      # generalised coordinates for hidden causes v (or \eta in biorxiv manuscript), but only using n-1

x = np.zeros((hidden_states, embed_orders_states))           # position
x1 = np.zeros((hidden_states, embed_orders_states))           # position
x2 = np.zeros((hidden_states, embed_orders_states))           # position
    
v = np.zeros((hidden_causes, embed_orders_causes))

def f(x, v):
    return np.dot(A, x) + np.dot(B, v)

x = 10 * np.random.rand(hidden_states, embed_orders_states) - 5
w = np.random.randn(iterations, hidden_states)

a = np.dot(C, w[0, :, None])
x1[:, 1:] = f(x[:, :-1], v) + np.dot(C, w[0, :, None])
x1[:, 0] += dt * x[:, 1]

print('x1 = ', x1)


# smaller test
AA = np.array([[0, 1], [-3, -17]])
AA_tilde = np.kron(np.eye(3), AA)
# AA_tilde2 = np.kron(AA, np.eye(3))

print(AA_tilde)
# print(AA_tilde2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       


Azzurro = np.random.randn(15, 4)
Azzurro[1::2,:] = 0


from autograd import grad
MAX_TEMPORAL_ORDERS = 20


def rho(h, phi):
    return np.exp(-.5*h**2/phi)

drho = [grad(rho,0)]
for i in range(1,MAX_TEMPORAL_ORDERS):
    drho.append(grad(drho[i-1]))

def dnrho(h, phi, degree):
    return drho[degree](h, phi)

def smoothnessMatrix(emb, ):
    h = 0.0                                                       # lag
    phi = .5
    embedding_orders = emb
    derivative_order = (embedding_orders-1)*2+1
    rho_tilde = np.zeros(derivative_order,)                     # array of autocorrelations
    S_inv = np.zeros((embedding_orders,embedding_orders))             # temporal roughness
    S = np.zeros((embedding_orders,embedding_orders))             # temporal smoothness

    rho_tilde[0] = rho(h, phi)
    for i in range(1, derivative_order):
        rho_tilde[i] = dnrho(h, phi, i-1)

    for i in range(embedding_orders):
        for j in range(embedding_orders):
            S_inv[i, j] = np.power(-1, (i*j))*rho_tilde[i+j]
    S = np.linalg.inv(S_inv)

    return S
    
    
S = smoothnessMatrix(embed_orders_states)

