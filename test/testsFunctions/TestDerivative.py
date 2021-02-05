import torch
import matplotlib.pyplot as plt


dt = .01
T = .1
iterations = int(T/dt)

x = 10*torch.ones(2,1, requires_grad=True)
a = torch.zeros(2,1, requires_grad=True)

x_history = 10*torch.randn(2,iterations, requires_grad=True)
a_history = torch.randn(2,iterations, requires_grad=True)

A = torch.tensor([[0., 1.], [-2., -1.]])
B = torch.tensor([[0., 0.], [0., 1.]])

def loss(x, a):
    # return torch.norm(x)
    # return .5 * x.t() @ x
    return .5 * (A @ x + B @ a).t() @ (A @ x + B @ a)

for i in range(iterations-1):
    print(i)

    dx = A @ x + B @ a

    with torch.no_grad():
        x = x + dt * dx
        x.requires_grad = True

    L = loss(x, a)

    L.backward(retain_graph=True)

    print(dx.grad)
    print(x.grad)

    # history
    x_history[:,i] = x.squeeze()
    a_history[:,i] = a.squeeze()


plt.figure()
plt.plot(x_history[0, :iterations-1].detach())

plt.figure()
plt.plot(x_history[1, :iterations-1].detach())

plt.show()