import torch

A = torch.zeros(12)
B = torch.arange(0.,12)

print(A)
print(B)

interval = 3
jumps = int(len(A)/interval)

for i in range(jumps):
    A[(i+1)*interval-1] = B[(i+1)*interval-1]

print(A)
print(B)
