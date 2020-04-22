from Functions import *

embedding_orders = 3
variables = 3

k = .25
phi = 2.
sigma_ddw = 1.
Sigma_w = np.array([[.25*sigma_ddw**2/(k**2), 0, 0], [0, .5*sigma_ddw**2/k, 0], [0, 0, sigma_ddw**2]])
Pi_w = np.linalg.pinv(Sigma_w)

w = Sigma_w @ np.random.randn(variables, 1)
w1 = (np.linalg.pinv(Sigma_w) @ w)[0,0]
S = np.linalg.pinv(smoothnessMatrix(8, phi))
print(np.array2string(S, formatter={'float_kind':'{0:.3f}'.format}))

w_tilde = np.kron(w, S)
w_tilde1 = np.kron(S, w)

w_tilde2 = np.kron(w1, S)
w_tilde3 = np.kron(S, w1)

sigma_w = 3.
pi_w = 1/sigma_w**2
pi_w_tilde = np.kron(pi_w, S)

sigma_w_tilde = np.kron(sigma_w, S)

a = 1