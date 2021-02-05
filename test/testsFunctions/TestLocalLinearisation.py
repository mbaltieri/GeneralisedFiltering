import torch
import matplotlib.pyplot as plt

## Euler-Maruyama

dt = .01
T = 10
iterations = int(T/dt)

simulations_n = 1

def F(x, v):
    return - 1/4 * x**4 + 1/2 * x ** 2 + 1/3 * v**3

# x = torch.zeros(iterations, simulations_n, requires_grad=True)
# v = torch.zeros(iterations, simulations_n, requires_grad=True)

# with torch.no_grad():
#     x[0] = torch.randn(1, simulations_n)



# for i in range(iterations-1):
#     print(i)

#     cost = F(x[i,:], v[i,:])
#     cost.backward()
#     dFdx = x.grad[i]

#     with torch.no_grad():
#         x[i+1,:] = x[i,:] + dt * dFdx

#         x.grad = None



# plt.figure()
# plt.plot(x.detach())


## Local-linearisation

dt = 1.
T = 10
iterations = int(T/dt)

def dFdx(x, v):
    return x + 0.
    # cost = F(x, v)
    # cost.backward()
    # return x.grad + 1

x = torch.zeros(iterations, simulations_n, requires_grad=True)
v = torch.zeros(iterations, simulations_n, requires_grad=True)

with torch.no_grad():
    x[0] = torch.randn(1, simulations_n)

for i in range(iterations-1):
    print(i)

    inputs = (x, v)

    cost = F(x[i,:], v[i,:])
    cost.backward()
    dx = x.grad[i,:]
    dv = v.grad[i,:]
    f = dFdx(dx, dv)

    inputs = (dx, dv)
    J = torch.autograd.functional.jacobian(dFdx, inputs)
    inputs = (x[i,:], v[i,:])
    H = torch.autograd.functional.hessian(F, inputs)

    # print(f)
    # print(J)
    # print(H)
    # print(H[0][0])

    dx = (torch.matrix_exp(dt * H[0][0]) - torch.eye(simulations_n)) @ H[0][0].inverse() @ f

    with torch.no_grad():
        x[i+1,:] = x[i,:] + dt * dx

        x.grad = None


plt.figure()
plt.plot(x.detach())


plt.show()