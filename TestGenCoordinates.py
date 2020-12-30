from functions import *
import torch
import matplotlib.pyplot as plt

# T = 100
# dt = 1
# iterations = int(T/dt)

# embedding_orders = 3
# variables = 3

# k = .25
# phi = 2.
# sigma_ddw = 1.
# Sigma_w = np.array([[.25*sigma_ddw**2/(k**2), 0, 0], [0, .5*sigma_ddw**2/k, 0], [0, 0, sigma_ddw**2]])
# Pi_w = np.linalg.pinv(Sigma_w)

# w = Sigma_w @ np.random.randn(variables, 1)
# w1 = (np.linalg.pinv(Sigma_w) @ w)[0,0]
# S = np.linalg.pinv(temporalPrecisionMatrix(8, phi))
# print(np.array2string(S, formatter={'float_kind':'{0:.3f}'.format}))

# w_tilde = np.kron(w, S)
# w_tilde1 = np.kron(S, w)

# w_tilde2 = np.kron(w1, S)
# w_tilde3 = np.kron(S, w1)

# sigma_w = 3.
# pi_w = 1/sigma_w**2
# pi_w_tilde = np.kron(pi_w, S)

# sigma_w_tilde = np.kron(sigma_w, S)

# a = 1













# # test spm_DEM_z
# noise = torch.randn(1, iterations)
# smoothened_noise = spm_DEM_z(1, 5., iterations, 1)

# P1 = toeplitz(np.exp(-np.arange(0, T, dt)**2/(2*.158**2)))
# t = np.arange(0,T,dt)
# sigma = 0.158
# P2 = toeplitz(np.exp(-t**2/(2*sigma**2)))
# # P = toeplitz(np.exp(-np.arange(0, iterations, 1)**2/(2*.15**2)))
# F = np.diag(1./np.sqrt(np.diag(np.dot(P1.T,P1))))
# # Make the smoothened noise:
# K = np.dot(P1,F)
# smoothened_noise2 = np.dot(noise,K)

# plt.figure()
# plt.plot(range(iterations), noise[0, :])
# plt.plot(range(iterations), smoothened_noise[0, :])
# plt.plot(range(iterations), smoothened_noise2[0, :])


# # np.random.seed(3)
# # Setting up the time data:
# dt = 0.05
# T = 5+dt
# N = int(round(T/dt))
# t = np.arange(0,T-dt,dt)
# # Desired covariance matrix (noise in Rˆ2):
# # note: this matrix must be symmetric
# Sw = np.matrix('1 0;0 1')
# # Generate white Gaussian noise sequences:
# n = Sw.shape[1] # dimension of noise
# L =sqrtm(Sw)  #Sqrtm method
# #L=cholesky(Sw, lower=True)  #Cholesky method
# w = np.dot(L,np.random.randn(n,N))
# w = np.random.randn(n,N)
# w = noise.numpy()
# # Plot the first white noise sequence:
# plt.figure()
# plt.plot(t,w.T[:,0],label='test')
# # Set up convolution matrix:
# sigma = 0.158
# P3 = toeplitz(np.exp(-t**2/(2*sigma**2)))
# F = np.diag(1./np.sqrt(np.diag(np.dot(P3.T,P3))))
# # Make the smoothened noise:
# K = np.dot(P3,F)
# ws = np.dot(w,K)
# plt.plot(t,ws.T[:,0]) # some plot versions plot expect data in same dimension, hence the ws.T to align with w
# plt.title('$\sigma=$ ' + str(sigma))
# plt.show()
















N = 2               # number of dimensions
genCoord = 4        # number of embedding orders
phi = 2.


# test spm_DEM_embed

dt = .01
T = 50
iterations = int(round(T/dt))

noiseNEW = spm_DEM_z(N, phi, T, dt)

noiseGENCOORD = torch.zeros(iterations, N*genCoord, genCoord)

for i in range(iterations):
    noiseGENCOORD[i,:,:] = spm_DEM_embed(noiseNEW, genCoord, i, dt=1.)

plt.figure()
for i in range(N*genCoord):
    for j in range(genCoord):
        plt.plot(noiseGENCOORD[:, i, j])

plt.figure()
for i in range(N):
    plt.plot(noiseNEW[i,:])
# print(noiseGENCOORD)

plt.show()