import torch

A = torch.tensor([[2., 3.], [3., 20.]])

B = torch.cholesky(A)

print(B)
print(B @ B.t())


def symsqrt(matrix):
    """Compute the square root of a positive definite matrix."""
    _, s, v = matrix.svd()
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


x = torch.randn(5, 10, 10).double()
x1 = x.transpose(-1, -2)
x = x @ x.transpose(-1, -2)
y = symsqrt(x)
print(torch.allclose(x, y @ y.transpose(-1, -2)))
x.requires_grad = True
torch.autograd.gradcheck(symsqrt, [x])
torch.autograd.gradgradcheck(symsqrt, [x])


B = symsqrt(A)

print(B)
print(B @ B.t())