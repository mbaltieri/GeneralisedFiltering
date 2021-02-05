import torch

A = torch.tensor([[0, 0., 0], [1, 0, 0], [0, 0, 0]])
B = torch.tensor([[1, 0., 0], [0, 0, 0], [0, 0, 0]])

print(A)
print(A.pinverse())

print(B)
print(B.pinverse())

print(A+B)
print((A+B).pinverse())