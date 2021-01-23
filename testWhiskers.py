import torch
import matplotlib.pyplot as plt
from layer import *
from integrationSchemes import dx_ll
import math

torch.set_printoptions(precision=10)

print("Device in use:", DEVICE)

# torch.manual_seed(42)

dt = 1
T = 100
iterations = int(T/dt)

e_n = 1                             # embedding dimension hidden states

### Wisker model

## generative process
omega2 = 0.5

A = torch.tensor([[0., 1], [-omega2, 0]])
B_a = torch.tensor([[0., 0.], [0., 1.]], device=DEVICE)
B_u = torch.tensor([[0., 0.], [0., 1.]], device=DEVICE)
F = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device=DEVICE)

sigma_z_log = torch.tensor([-4.], device=DEVICE)                                     # log-precision
sigma_z = torch.exp(sigma_z_log)
Sigma_z = torch.tensor([[sigma_z, 0., 0.], [0., sigma_z, 0.], 
            [0., 0, sigma_z]], device=DEVICE)

sigma_w_log = torch.tensor([-4.], device=DEVICE)                                     # log-precision
sigma_w = torch.exp(sigma_w_log)
Sigma_w = torch.tensor([[0., 0.], [0., sigma_w]], device=DEVICE)



## generative model
alpha = torch.exp(torch.tensor([1.]))
alpha2 = torch.exp(torch.tensor([1.]))
beta = torch.exp(torch.tensor([1.]))

A_gm = torch.tensor([[0., 1.], [-alpha, -alpha2]], device=DEVICE)                   # state transition matrix
# A_gm = A
B_u_gm = torch.tensor([[0, 0.], [0., 1]], device=DEVICE)                      # input matrix (dynamics)
F_gm = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device=DEVICE)      # observation matrix

sigma_z_log_GM = torch.tensor([-16.], device=DEVICE)                                  # log-precision
sigma_z_GM = torch.exp(sigma_z_log_GM)
Sigma_z_GM = torch.tensor([[sigma_z_GM, 0., 0.], [0., sigma_z_GM, 0.], 
            [0., 0, sigma_z_GM]], device=DEVICE)                                    # TODO: tensor type deault is 'int64' when no dot is used, but need float for pseudo-inverse

sigma_w_log_GM = torch.tensor([-8.], device=DEVICE)                                 # log-precision
sigma_w_GM = torch.exp(sigma_w_log_GM)
Sigma_w_GM = torch.tensor([[sigma_w_GM, 0.], [0., sigma_w_GM]], device=DEVICE)

sigma_v_log_GM = torch.tensor([16.], device=DEVICE)                                # log-precision
sigma_v_GM = torch.exp(sigma_v_log_GM)
Sigma_v_GM = torch.tensor([[sigma_v_GM, 0., 0.], [0., sigma_v_GM, 0.], 
            [0., 0, sigma_v_GM]], device=DEVICE)

dyda = torch.tensor([[0., 0., 0.], [0., 1., 1.], [0., 0., 0.]], device=DEVICE)
eta_u = torch.tensor([[0.], [0.], [0.]], device=DEVICE)                             # desired state


GP = layer('GP', T, dt, A=A, F=F, Sigma_w=Sigma_w, Sigma_z=Sigma_z, e_n=e_n, B_a=B_a, B_u=B_u)
GM = layer('GM', T, dt, A=A_gm, F=F_gm, Sigma_w=Sigma_w_GM, Sigma_z=Sigma_z_GM, Sigma_v=Sigma_v_GM, e_n=e_n, dyda=dyda, B_u=B_u_gm, eta_u=eta_u)


for i in range(0,iterations-1):
    print(i)

    GP.step(i)
    GM.setObservations(GP.y)

    # # calculate loss
    F = GM.free_energy(GM.y, GM.x, GM.u, i)

    GM.inferencestep(i)
    # GP.a = GM.a
    
    if (i > iterations/3) and (i < 2*iterations/3):
        GP.A[1, 1] = -.02
        GP.u[1] = 30*(i-iterations/3)/iterations
    if (i >= 2*iterations/3):
        GP.A[1, 1] = 0.

    GP.saveHistoryVariables(i)
    GM.saveHistoryVariables(i)

plt.figure()
plt.plot(GM.y_history[:-1,0].detach(), GM.y_history[:-1,1].detach(), 'b')
plt.plot(GM.y_history[0,0].detach(), GM.y_history[0,1].detach(), 'bo')
plt.plot(GM.x_history[:-1,0].detach(), GM.x_history[:-1,1].detach(), 'r')
plt.plot(GM.x_history[0,0].detach(), GM.x_history[0,1].detach(), 'ro')

figU = plt.figure(figsize=(15,5))
ax1 = figU.add_subplot(131)
ax2 = figU.add_subplot(132)
ax3 = figU.add_subplot(133)

ax1.plot(range(iterations-1), GP.u_history[:-1,0].detach(), 'b')
ax1.plot(range(iterations-1), GM.u_history[:-1,0].detach(), 'r')

ax2.plot(range(iterations-1), GP.u_history[:-1,1].detach(), 'b')
ax2.plot(range(iterations-1), GM.u_history[:-1,1].detach(), 'r')

ax3.plot(range(iterations-1), GP.u_history[:-1,2].detach(), 'b')
ax3.plot(range(iterations-1), GM.u_history[:-1,2].detach(), 'r')

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.plot(range(iterations-1), GM.y_history[:-1,0].detach(), 'b')
ax1.plot(range(iterations-1), GM.x_history[:-1,0].detach(), 'r')
ax1.plot(range(iterations-1), GP.x_history[:-1,0].detach(), 'k')
ax1.plot(range(iterations-1), GM.u_history[:-1,0].detach(), 'g')

ax2.plot(range(iterations-1), GM.y_history[:-1,1].detach(), 'b')
ax2.plot(range(iterations-1), GM.x_history[:-1,1].detach(), 'r')
ax2.plot(range(iterations-1), GP.x_history[:-1,1].detach(), 'k')
ax2.plot(range(iterations-1), GM.u_history[:-1,1].detach(), 'g')

ax3.plot(range(iterations-1), GM.y_history[:-1,2].detach(), 'b')
ax3.plot(range(iterations-1), GM.x_history[:-1,2].detach(), 'r')
ax3.plot(range(iterations-1), GP.x_history[:-1,2].detach(), 'k')
ax3.plot(range(iterations-1), GM.u_history[:-1,2].detach(), 'g')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(range(iterations-1), GP.a_history[:-1,0].detach(), 'b')
ax1.plot(range(iterations-1), GP.a_history[:-1,1].detach(), 'r')
ax1.plot(range(iterations-1), GP.a_history[:-1,2].detach(), 'k')

fig2 = plt.figure(figsize=(15,5))
plt.title('Prediction errors')
ax1 = fig2.add_subplot(131)
ax2 = fig2.add_subplot(132)
ax3 = fig2.add_subplot(133)

ax1.plot(range(iterations-1), GM.eps_v_history[:-1,0].detach(), 'b')
ax1.plot(range(iterations-1), GM.eps_v_history[:-1,1].detach(), 'r')
ax1.plot(range(iterations-1), GM.eps_v_history[:-1,2].detach(), 'k')

ax2.plot(range(iterations-1), GM.eps_x_history[:-1,0].detach(), 'b')
ax2.plot(range(iterations-1), GM.eps_x_history[:-1,1].detach(), 'r')
ax2.plot(range(iterations-1), GM.eps_x_history[:-1,2].detach(), 'k')

ax3.plot(range(iterations-1), GM.eps_eta_history[:-1,0].detach(), 'b')
ax3.plot(range(iterations-1), GM.eps_eta_history[:-1,1].detach(), 'r')
ax3.plot(range(iterations-1), GM.eps_eta_history[:-1,2].detach(), 'k')


fig3 = plt.figure(figsize=(15,5))
plt.title('Weighted prediction errors')

ax1 = fig3.add_subplot(131)
ax2 = fig3.add_subplot(132)
ax3 = fig3.add_subplot(133)

ax1.plot(range(iterations-1), GM.xi_v_history[:-1,0].detach(), 'b')
ax1.plot(range(iterations-1), GM.xi_v_history[:-1,1].detach(), 'r')
ax1.plot(range(iterations-1), GM.xi_v_history[:-1,2].detach(), 'k')

ax2.plot(range(iterations-1), GM.xi_x_history[:-1,0].detach(), 'b')
ax2.plot(range(iterations-1), GM.xi_x_history[:-1,1].detach(), 'r')
ax2.plot(range(iterations-1), GM.xi_x_history[:-1,2].detach(), 'k')

ax3.plot(range(iterations-1), GM.xi_eta_history[:-1,0].detach(), 'b')
ax3.plot(range(iterations-1), GM.xi_eta_history[:-1,1].detach(), 'r')
ax3.plot(range(iterations-1), GM.xi_eta_history[:-1,2].detach(), 'k')

plt.show()