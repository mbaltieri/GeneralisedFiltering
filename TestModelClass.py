from layer import *
import functions
import matplotlib.pyplot as plt

# torch.autograd.set_detect_anomaly(True)

dt = .5
T = 3
iterations = int(T/dt)

l = 3                               # number of layers

m = 2                               # observations dimension
n = 2                               # hidden states dimension
r = 2                               # inputs dimension
p = 6                               # parameters dimension
h = 3                               # hyperparameters dimension

e_n = 1                             # embedding dimension hidden states
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

GP = layer(T, dt, A=A, F=F, Sigma_w=Sigma_w, Sigma_z=Sigma_z, e_n=e_n, history=iterations)            # TODO: At the moment simulating only sequencies with Brownian motion, 
                                                                            # since Gaussian autocorrelations introduce negative covariances that 
                                                                            # can't truly be simulated (as far as I know), Karl has a way to build 
                                                                            # generalised sequencies out of this so check spm
GM = layer(T, dt, A=A, F=F, Sigma_w=Sigma_w_GM, Sigma_z=Sigma_z, e_n=e_n, history=iterations)

for i in range(iterations-1):
    print(i)

    GP.step(i)
    # GM.setObservations(GP.y.detach())
    GM.setObservations(GP.y)

    # GM.generalisedCoordinates()
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
    # GP.save_history(i)


fig = plt.figure()
ax1 = fig.add_subplot(151)
ax2 = fig.add_subplot(152)
ax3 = fig.add_subplot(153)
ax4 = fig.add_subplot(154)
ax5 = fig.add_subplot(155)

ax1.plot(range(iterations-1), GP.y_history[:-1,0,0].detach(), 'b')
ax1.plot(range(iterations-1), GM.x_history[:-1,0,0].detach(), 'r')

ax2.plot(range(iterations-1), GP.w_history[:-1,0,0].detach(), 'b')
ax2.plot(range(iterations-1), GP.w_history[:-1,1,0].detach(), 'r')

ax3.plot(range(iterations-1), GP.wSmoothened[0,:-1].detach(), 'b')
ax3.plot(range(iterations-1), GP.wSmoothened[1,:-1].detach(), 'r')

ax4.plot(range(iterations-1), GP.z_history[:-1,0,0].detach(), 'b')
ax4.plot(range(iterations-1), GP.z_history[:-1,1,0].detach(), 'r')
ax4.plot(range(iterations-1), GP.z_history[:-1,2,0].detach(), 'g')
ax4.plot(range(iterations-1), GP.z_history[:-1,3,0].detach(), 'k')

ax5.plot(range(iterations-1), GP.zSmoothened[0,:-1].detach(), 'b')
ax5.plot(range(iterations-1), GP.zSmoothened[1,:-1].detach(), 'r')
ax5.plot(range(iterations-1), GP.zSmoothened[2,:-1].detach(), 'g')
ax5.plot(range(iterations-1), GP.zSmoothened[3,:-1].detach(), 'k')

plt.show()