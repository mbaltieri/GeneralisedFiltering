import torch

x = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])

print(x)

y = x.view(2,5)[:,:-1].t()

print(y)

# z = y.view(8,1)
y = y.t().flatten().unsqueeze(1)

print(y)

y[1] = 100.

# print(z)
print(y)

y = y.squeeze().unflatten(0, (2, 4)).t()

print(y)

# y[1, 2] = 100.

# print(y)
# print(x)



# a = torch.tensor([[1, 2, 3., 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])

# print(a)

# b = a[:-1,:].t().flatten().unsqueeze(0).t()

# b[7, 0] = 100

# print(b)

# print(a)