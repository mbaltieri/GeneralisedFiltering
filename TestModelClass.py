from layer import *
import functions
import matplotlib.pyplot as plt

# torch.autograd.set_detect_anomaly(True)

dt = .01
T = 30
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
F = torch.tensor([[1, 0], [0, 1]], device=DEVICE)

Sigma_w = torch.tensor([[0., 0.], [0., .0001]], device=DEVICE)
Sigma_z = torch.tensor([[2., 0., 0., 0.], [0., 1., 0., 0.], [0., 0, .5, 0.], [0., 0., 0., .25]], device=DEVICE)                       # TODO: tensor type deault is 'int64' but need float for pseudo-inverse
Sigma_z = torch.tensor([[.1, 0], [0, .1]], device=DEVICE)

Sigma_w_GM = torch.tensor([[.1, 0.], [0., .1]], device=DEVICE)

dyda = []

GP = layer(T, dt, A=A, F=F, Sigma_w=Sigma_w_GM, Sigma_z=Sigma_z, e_n=e_n)
GM = layer(T, dt, A=A, F=F, Sigma_w=Sigma_w_GM, Sigma_z=Sigma_z, e_n=e_n)

learning_rate = 5e-3
learning_rate = 0.03

for i in range(iterations-1):
    print(i)

    GP.step(i)
    # GM.setObservations(GP.y.detach())
    GM.setObservations(GP.y)

    F = GM.free_energy(i)

    # Use autograd to compute the backward pass.
    F.backward()

    # Update weights using gradient descent
    dFdy = GM.y.grad
    dFdx = GM.x.grad
    dFdv = GM.v.grad
    with torch.no_grad():
        GM.x -= learning_rate * dt * dFdx
        GM.v -= learning_rate * dt * dFdv
    
    # GP.setActions(a)

        # Manually zero the gradients after updating weights
        GM.x.grad = None
        GM.v.grad = None
    

    
    GM.save_history(i)

plt.figure()
plt.plot(GM.F_history[:-1].detach())

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# ax1 = fig.add_subplot(251)
# ax2 = fig.add_subplot(252)
# ax3 = fig.add_subplot(253)
# ax4 = fig.add_subplot(254)
# ax5 = fig.add_subplot(255)

# ax7 = fig.add_subplot(257)
# ax9 = fig.add_subplot(259)

ax1.plot(range(iterations-1), GM.y_history[:-1,0].detach(), 'b')
ax1.plot(range(iterations-1), GM.x_history[:-1,0].detach(), 'r')
ax1.plot(range(iterations-1), GP.x_history[:-1,0].detach(), 'k')

ax2.plot(range(iterations-1), GM.y_history[:-1,1].detach(), 'b')
ax2.plot(range(iterations-1), GM.x_history[:-1,1].detach(), 'r')
ax2.plot(range(iterations-1), GP.x_history[:-1,1].detach(), 'k')

# ax2.plot(range(iterations-1), GP.w_history[:-1,0,0].detach(), 'b')
# ax2.plot(range(iterations-1), GP.w_history[:-1,1,0].detach(), 'r')

# ax7.plot(range(iterations-1), GP.w_history[:-1,2,0].detach(), 'b')
# ax7.plot(range(iterations-1), GP.w_history[:-1,3,0].detach(), 'r')

# ax3.plot(range(iterations-1), GP.wSmoothened[0,:-1].detach(), 'b')
# ax3.plot(range(iterations-1), GP.wSmoothened[1,:-1].detach(), 'r')

# ax4.plot(range(iterations-1), GP.z_history[:-1,0,0].detach(), 'b')
# ax4.plot(range(iterations-1), GP.z_history[:-1,1,0].detach(), 'r')
# ax4.plot(range(iterations-1), GP.z_history[:-1,2,0].detach(), 'g')
# ax4.plot(range(iterations-1), GP.z_history[:-1,3,0].detach(), 'k')

# ax9.plot(range(iterations-1), GP.z_history[:-1,4,0].detach(), 'b')
# ax9.plot(range(iterations-1), GP.z_history[:-1,5,0].detach(), 'r')
# ax9.plot(range(iterations-1), GP.z_history[:-1,6,0].detach(), 'g')
# ax9.plot(range(iterations-1), GP.z_history[:-1,7,0].detach(), 'k')

# ax5.plot(range(iterations-1), GP.zSmoothened[0,:-1].detach(), 'b')
# ax5.plot(range(iterations-1), GP.zSmoothened[1,:-1].detach(), 'r')
# ax5.plot(range(iterations-1), GP.zSmoothened[2,:-1].detach(), 'g')
# ax5.plot(range(iterations-1), GP.zSmoothened[3,:-1].detach(), 'k')

plt.show()