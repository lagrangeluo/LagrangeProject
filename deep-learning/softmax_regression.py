import torch
from IPython import display
from d2l import torch as d2l

# 获取mnist数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# mnist数据图片格式为24X24，训练时将向量摊平为784
num_inputs = 784
num_outputs = 10

# 设置权重和偏置
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制