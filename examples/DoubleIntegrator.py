import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import src.functions
from src.layer import *
from src.integrationschemes import dx_ll
from src.globals import DEVICE

import matplotlib.pyplot as plt

# torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=10)
# print(torch.__version__)

print("Device in use:", DEVICE)

torch.manual_seed(42)

# if torch.cuda.is_available():
#     torch.set_default_tensor_type(torch.cuda.DoubleTensor)
# else:
#     torch.set_default_tensor_type(torch.DoubleTensor)

# torch.set_default_dtype(torch.float64)                                                                # TODO: tensor type deault is 'int64' when no dot is used, but need float for pseudo-inverse, can we do without specifying dtype all the time?

dt = 1
T = 50
iterations = int(T/dt)

l = 3                               # number of layers

e_n = 1                             # embedding dimension hidden states
e_r = 0                             # embedding dimension inputs
e_p = 0                             # embedding dimension parameters
e_h = 0                             # embedding dimension hyperparameters

### Test double integrator

## generative process

A = torch.tensor([[0, 1], [0, 0]], dtype=torch.float64, device=DEVICE)                                  # state transition matrix
# A = torch.tensor([[0, 1], [-2, -1]], dtype=torch.float64, device=DEVICE)                              # state transition matrix
B_a = torch.tensor([[0, 0], [0, 1]], dtype=torch.float64, device=DEVICE)                                # input matrix (dynamics)
F = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64, device=DEVICE)                 # observation matrix

sigma_z_log = torch.tensor([0], dtype=torch.float64, device=DEVICE)                                     # log-precision
sigma_z = torch.exp(sigma_z_log)
Sigma_z = torch.tensor([[sigma_z, 0, 0], [0, sigma_z, 0], 
            [0, 0, sigma_z]], dtype=torch.float64, device=DEVICE)

sigma_w_log = torch.tensor([0], dtype=torch.float64, device=DEVICE)                                     # log-precision
sigma_w = torch.exp(sigma_w_log)
Sigma_w = torch.tensor([[0, 0], [0, sigma_w]], dtype=torch.float64, device=DEVICE)

def g(x, u, a, F, G):
    # TODO: generalise this to include nonlinear treatments
    try:
        return F @ x + G @ u
    except RuntimeError:
        print("Dimensions don't match!")
        return


## generative model
alpha = torch.exp(torch.tensor([1], dtype=torch.float64))
alpha2 = torch.exp(torch.tensor([1], dtype=torch.float64))
beta = torch.exp(torch.tensor([1], dtype=torch.float64))

A_gm = torch.tensor([[0, 1], [-alpha, -alpha2]], dtype=torch.float64, device=DEVICE)                    # state transition matrix
B_u_gm = torch.tensor([[beta, 0], [0, beta]], device=DEVICE)                                            # input matrix (dynamics)
F_gm = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64, device=DEVICE)              # observation matrix

sigma_z_log_GM = torch.tensor([2.5], dtype=torch.float64, device=DEVICE)                                # log-precision
sigma_z_GM = torch.exp(sigma_z_log_GM)
Sigma_z_GM = torch.tensor([[sigma_z_GM, 0, 0], [0, sigma_z_GM, 0], 
            [0, 0, sigma_z_GM]], dtype=torch.float64, device=DEVICE)

sigma_w_log_GM = torch.tensor([-8], dtype=torch.float64, device=DEVICE)                                 # log-precision
sigma_w_GM = torch.exp(sigma_w_log_GM)
Sigma_w_GM = torch.tensor([[sigma_w_GM, 0], [0, sigma_w_GM]], dtype=torch.float64, device=DEVICE)

sigma_v_log_GM = torch.tensor([-16], dtype=torch.float64, device=DEVICE)                                # log-precision
sigma_v_GM = torch.exp(sigma_v_log_GM)
Sigma_v_GM = torch.tensor([[sigma_v_GM, 0, 0], [0, sigma_v_GM, 0], 
            [0, 0, sigma_v_GM]], dtype=torch.float64, device=DEVICE)

dyda = 10*torch.tensor([[0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float64, device=DEVICE)           # in his code, Karl uses dyda as derived from the generative process
eta_u = torch.tensor([[0], [0], [0]], dtype=torch.float64, device=DEVICE)                               # desired state

## create models
GP = layer('GP', T, dt, A=A, F=F, g=g, Sigma_w=Sigma_w, Sigma_z=Sigma_z, e_n=e_n, B_a=B_a)
GM = layer('GM', T, dt, A=A_gm, F=F_gm, g=g, Sigma_w=Sigma_w_GM, Sigma_z=Sigma_z_GM, Sigma_v=Sigma_v_GM, e_n=e_n, dyda=dyda, B_u=B_u_gm, eta_u=eta_u)

for i in range(0,iterations-1):
    print(i)

    GP.step(i)
    GM.setObservations(GP.y)

    # calculate loss
    F = GM.free_energy(GM.y, GM.x, GM.u, i)

    GM.inferencestep(i)
    GP.a = GM.a

    GP.saveHistoryVariables(i)
    GM.saveHistoryVariables(i)

plt.figure()
plt.plot(GM.y_history[:-1,0].detach(), GM.y_history[:-1,1].detach(), 'b')
plt.plot(GM.y_history[0,0].detach(), GM.y_history[0,1].detach(), 'bo')
plt.plot(GM.x_history[:-1,0].detach(), GM.x_history[:-1,1].detach(), 'r')
plt.plot(GM.x_history[0,0].detach(), GM.x_history[0,1].detach(), 'ro')

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