from layer import *
import functions
import matplotlib.pyplot as plt

# torch.autograd.set_detect_anomaly(True)

dt = .05
T = 50
iterations = int(T/dt)

l = 3                               # number of layers

m = 2                               # observations dimension
n = 2                               # hidden states dimension
r = 2                               # inputs dimension
p = 6                               # parameters dimension
h = 3                               # hyperparameters dimension

e_n = 3                             # embedding dimension hidden states
e_r = 0                             # embedding dimension inputs
e_p = 0                             # embedding dimension parameters
e_h = 0                             # embedding dimension hyperparameters

delta = -1.50                                                # parameters simulating a simple mass-spring system as the environment
epsilon = -10.0

A = torch.tensor([[0, 1], [delta, epsilon]], device=DEVICE)
F = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]], device=DEVICE)

Sigma_w = torch.tensor([[0., 0.], [0., .1]], device=DEVICE)
Sigma_z = torch.tensor([[2., 0., 0., 0.], [0., 1., 0., 0.], [0., 0, .5, 0.], [0., 0., 0., .25]], device=DEVICE)                       # TODO: tensor type deault is 'int64' but need float for pseudo-inverse

Sigma_w_GM = torch.tensor([[.1, 0.], [0., .1]], device=DEVICE)

GP = layer(T, dt, A=A, F=F, Sigma_w=Sigma_w, Sigma_z=Sigma_z, e_n=0, history=iterations)            # TODO: At the moment simulating only sequencies with Brownian motion, 
                                                                            # since Gaussian autocorrelations introduce negative covariances that 
                                                                            # can't truly be simulated (as far as I know), Karl has a way to build 
                                                                            # generalised sequencies out of this so check spm
GM = layer(T, dt, A=A, F=F, Sigma_w=Sigma_w_GM, Sigma_z=Sigma_z, e_n=0, history=iterations)

for i in range(iterations):
    print(i)

    GP.step(i)
    GM.setObservations(GP.y.detach())

    F = GM.free_energy(i)
    
    # retain gradients for intermediate variables
    # TODO: find a better way to deal with this
    # GM.x.retain_grad()
    # GM.v.retain_grad()


    # F.backward(retain_graph=True)                             # TODO: is retain_graph=True necessary? Or even correct?
    F.backward()

    # dFdx = GM.x.grad
    # dFdv = GM.v.grad

    # # update variables
    # GM.x += dt * dFdx
    # GM.v += dt * dFdv
    
    # GM.save_history(i)
    GP.save_history(i)

    
print(GM.F_history)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(range(iterations), GP.x_history[:,0,0].detach(), 'b')
# ax1.plot(range(iterations), GM.x_history[:,0,0].detach(), 'r')

ax2.plot(range(iterations), GP.w_history[:,0,0].detach(), 'b')
ax2.plot(range(iterations), GP.w_history[:,1,0].detach(), 'r')

plt.show()