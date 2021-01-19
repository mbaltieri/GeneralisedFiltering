import functions
import matplotlib.pyplot as plt
from layer import *
from integrationSchemes import dx_ll

# torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=10)
# print(torch.__version__)

print("Device in use:", DEVICE)

# torch.manual_seed(42)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)

dt = 1
T = 20
iterations = int(T/dt)

l = 3                               # number of layers

# m = 2                               # observations dimension
# n = 2                               # hidden states dimension
# r = 2                               # inputs dimension
# p = 6                               # parameters dimension
# h = 3                               # hyperparameters dimension

e_n = 1                             # embedding dimension hidden states
e_r = 0                             # embedding dimension inputs
e_p = 0                             # embedding dimension parameters
e_h = 0                             # embedding dimension hyperparameters

### Test double integrator

## generative process

A = torch.tensor([[0., 1.], [0., 0.]], device=DEVICE)                 # state transition matrix
# A = torch.tensor([[0., 1.], [-2., -1.]], device=DEVICE)                 # state transition matrix
# A = torch.tensor([[0., 1., 0.], [-2, -2, 0.], [0., 0., 0.]], device=DEVICE)
B_a = torch.tensor([[0., 0.], [0., 1.]], device=DEVICE)               # input matrix (dynamics)
F = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device=DEVICE)               # observation matrix

sigma_z_log = torch.tensor([-8.], device=DEVICE)                                     # log-precision
sigma_z = torch.exp(sigma_z_log)
Sigma_z = torch.tensor([[sigma_z, 0., 0.], [0., sigma_z, 0.], 
            [0., 0, 0.]], device=DEVICE)                                       # TODO: tensor type deault is 'int64' when no dot is used, but need float for pseudo-inverse

sigma_w_log = torch.tensor([-8.], device=DEVICE)                                     # log-precision
sigma_w = torch.exp(sigma_w_log)
Sigma_w = torch.tensor([[0., 0.], [0., sigma_w]], device=DEVICE)


## generative model
alpha = torch.exp(torch.tensor([1.]))
alpha2 = torch.exp(torch.tensor([1.]))
beta = torch.exp(torch.tensor([1.]))

A_gm = torch.tensor([[0., 1.], [-alpha, -alpha2]], device=DEVICE)    # state transition matrix
B_u_gm = torch.tensor([[beta, 0.], [0., beta]], device=DEVICE)    # input matrix (dynamics)
F_gm = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]], device=DEVICE)               # observation matrix

sigma_z_log_GM = torch.tensor([6.], device=DEVICE)                                  # log-precision
sigma_z_GM = torch.exp(sigma_z_log_GM)
Sigma_z_GM = torch.tensor([[sigma_z_GM, 0., 0.], [0., sigma_z_GM, 0.], 
            [0., 0, sigma_z_GM]], device=DEVICE)                                    # TODO: tensor type deault is 'int64' when no dot is used, but need float for pseudo-inverse

sigma_w_log_GM = torch.tensor([-4.], device=DEVICE)                                 # log-precision
sigma_w_GM = torch.exp(sigma_w_log_GM)
Sigma_w_GM = torch.tensor([[sigma_w_GM, 0.], [0., sigma_w_GM]], device=DEVICE)

sigma_v_log_GM = torch.tensor([-8.], device=DEVICE)                                 # log-precision
sigma_v_GM = torch.exp(sigma_v_log_GM)
Sigma_v_GM = torch.tensor([[sigma_v_GM, 0., 0.], [0., sigma_v_GM, 0.], 
            [0., 0, sigma_v_GM]], device=DEVICE)

dyda = torch.exp(torch.tensor([8.]))*torch.tensor([[0., 0., 0.], [1., 1., 0.], [0., 0., 0.]], device=DEVICE)
eta_u = torch.tensor([[0.], [0.], [0.]], device=DEVICE)                            # desired state

## create models
GP = layer('GP', T, dt, A=A, F=F, Sigma_w=Sigma_w, Sigma_z=Sigma_z, e_n=e_n, B_a=B_a)
GM = layer('GM', T, dt, A=A_gm, F=F_gm, Sigma_w=Sigma_w_GM, Sigma_z=Sigma_z_GM, Sigma_v=Sigma_v_GM, e_n=e_n, dyda=dyda, B_u=B_u_gm, eta_u=eta_u)

for i in range(0,iterations-1):
    print(i)

    GP.step(i)
    GM.setObservations(GP.y)

    # calculate loss
    F = GM.free_energy(GM.y, GM.x, GM.u, i)

    # Use autograd to compute the backward pass        
    # F.backward()                                        # not feasible if we need Jacobians/Hessians later

    inputs = (GM.y, GM.x, GM.u)

    # Update weights using gradient descent
    # dFdy = GM.y.grad
    # dFdx = GM.x.grad
    # dFdu = GM.u.grad
    # dFda = GM.dyda @ dFdy

    dF = torch.autograd.functional.jacobian(lambda x, u, a, i=i: GM.free_energy(x, u, a, i), inputs)
    dFdy = dF[0].squeeze().unsqueeze(1)
    dFdx = dF[1].squeeze().unsqueeze(1)
    dFdu = dF[2].squeeze().unsqueeze(1)
    dFda = GM.dyda @ dFdy
    # print(dFda)

    
    J = torch.autograd.functional.hessian(lambda y, x, v: GM.free_energy(y, x, v, i=i), inputs)                         # TODO: wait for a decent implementation of 'hessian' and 'jacobian' on all inputs similar to backward
    J_y = J[0][0].squeeze()
    J_x = J[1][1].squeeze()                                                                                       # (at the moment both functions rely on grad, which requires specifying inputs). If not, to save some 
    J_u = J[2][2].squeeze()                                                                                       # time, might want to switch backward --> grad and than take jacobian of grad

    # print(J_y)
    # ab = kronecker(torch.eye(e_n+1), dyda)
    # print(ab)
    # J_a = kronecker(torch.eye(e_n+1), dyda).t() @ J_y @ kronecker(torch.eye(e_n+1), dyda)
    # print(J_a)
    J_a = J_y @ kronecker(torch.eye(e_n+1), dyda)**2
    # print(J_a)

    # # print('J_x: ', J_x)
    # # print('J_u: ', J_u)
    # # print('expm(J_x): ', torch.matrix_exp(dt * -J_x))

    with torch.no_grad():
        # integrate using Euler-Maruyama
        # GM.x = GM.x + dt * (Diff(GM.x, GM.sim, GM.e_sim+1) - dFdx)
        # GM.u = GM.u + dt * (Diff(GM.u, GM.sim, GM.e_sim+1) - dFdu)
        # GP.a = GP.a + dt * (- dFda)

        # integrate using Local-linearisation
        dx = dx_ll(dt, -J_x, (Diff(GM.x, GM.sim+1, GM.e_sim+1) - dFdx))                                          # NB: J --> - J since this is a minimisation of F, unlike the maximisation of -F in DEM and HMB
        du = dx_ll(dt, -J_u, (Diff(GM.u, GM.sim+1, GM.e_sim+1) - dFdu))                                          # NB: J --> - J since this is a minimisation of F, unlike the maximisation of -F in DEM and HMB
        da = 1000*dx_ll(dt, -J_a, - dFda)

        # print(da)

        GM.x = GM.x + dt * dx
        GM.u = GM.u + dt * du
        GP.a = GP.a + dt * da
        # print(GP.a)
        # GP.a[1,0] = 50*torch.sin(torch.tensor([2*math.pi*i/100]))
        # print(GP.a)

        # print(100*torch.sin(torch.tensor([0, 2*math.pi*i, 0])).unsqueeze(1))
        # print(GP.a)
    
        # Manually zero the gradients after updating weights
        GM.y.grad = None
        GM.x.grad = None
        GM.u.grad = None

        # GM.y.requires_grad = True
        GM.x.requires_grad = True                                               # FIXME: Definitely ugly, find a better way to deal with this (earlier, we had updates of the form GM.x[i+1] = GM.x[i] + ... 
        GM.u.requires_grad = True                                               # but this requires "i" as a parameter of the loss funciton, and "torch.autograd.functional.hessian" accepts only tensors as inputs)
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