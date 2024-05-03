import torch

x = torch.tensor(1.0, requires_grad = True)
a, b = 3,5

y = a * x + b
print(y)

# 微分する
y.backward()

print(x.grad)

v = torch.tensor(1.0, requires_grad = True)
w = torch.tensor(1.0, requires_grad = True)

a,b,c = 4,6,1

y2 = a * v + b * w + c

y2.backward()
print(v.grad)
print(w.grad)