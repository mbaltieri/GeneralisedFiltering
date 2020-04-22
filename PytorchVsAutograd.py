import torch
import autograd.numpy as np
from autograd import grad

# create a pytorch tensor and numpy array, same size (1x3)
# make sure that the tensor records the changes to calculate the gradient later
a = 4 * torch.ones((3), requires_grad=True)
a_np = 4 * np.ones(3)

# apply some algebraic operations to both
b = a/2
b_np = a_np/2

# define a function that applies some algebraic operations to both (in simple cases like 
# this one, the code is the same for both), vector output
def f_vector(x):
    return x**3

# parse the two objects through the function, giving a vector output
c = f_vector(b)
c_np = f_vector(b_np)

# check how they look
print('c = ', c)
print('c_cn = ', c_np)

# trying to find the gradient, none of these work, since the function returns a vector,
# not a scalar; backward can be generalised for the jacobian if necessary
# grad_f = grad(f, 0)
# d = grad_f(c_cp)
# d = c.backward()

# define a function that returns a scalar (pytorch case)
def f_scalar(y):
    return torch.dot(y, torch.tensor([0, 1.5, 0]))

# define a function that returns a scalar (numpy case)
def f_scalar_np(y):
    return np.dot(y, np.array([0, 1.5, 0]))

# parse the two objects through the new function, giving a scalar output
e = f_scalar(c)
e_np = f_scalar_np(c_np)

# check how they look
print('e = ', e)
print('e_np = ', e_np)

# trying to find the gradient
e.backward()
f = c.grad

grad_f = grad(f_scalar_np, 0)
f_np = grad_f(c_np)

print('f = ', f)
print('f_np = ', f_np)

# the gradient for pytorch is 'None' because none of these are leaf nodes, in the example below, x is leaf
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()
print(x.grad)