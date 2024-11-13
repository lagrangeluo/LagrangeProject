import torch
import random
from d2l import torch as d2l

x = torch.arange(12)

print("tensor shape: ",x.shape)
print("tensor size: ",x.numel())

X = x.reshape(3, 4)
y=torch.zeros(2,3,4)
print(y)
randon = torch.randn(3, 4)
#print(randon)

#caculate
x = torch.tensor([1, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)  # **运算符是求幂运算

#concatenate
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))
print(X == Y)
print(X.sum())
print(X[0:4])

#gradient
x = torch.arange(4.0,requires_grad=True)
y = 2 * torch.dot(x, x)
y.backward()
print(x.grad)

#线性回归模型训练
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)