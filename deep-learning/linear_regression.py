import random
import torch
import time
import matplotlib.pyplot as plt
from d2l import torch as d2l

import numpy as np
from torch.utils import data
from torch import nn
# 生成数据集
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 生成均值为0，标准差为1的正态分布样本作为特征矩阵X
    # X的维度为(num_examples, len(w))，即生成num_examples行，每行有len(w)个特征

    y = torch.matmul(X, w) + b
    # 计算线性模型 y = Xw + b

    y += torch.normal(0, 0.01, y.shape)
    # 在标签y中添加标准差为0.01的高斯噪声，模拟真实数据中的噪声干扰

    return X, y.reshape((-1, 1))
    # 返回生成的特征X和标签y，y被reshape成列向量形式

# 打乱数据集，并返回一个batch_size批次大小的小型数据
def data_iter(batch_size, features, labels):
    # 获取特征数据的样本总数
    num_examples = len(features)
    # 生成一个包含所有样本索引的列表
    indices = list(range(num_examples))  
    # 将样本索引随机打乱，确保样本是随机读取的
    random.shuffle(indices)
    # 按照 batch_size 大小逐步生成批次数据
    for i in range(0, num_examples, batch_size):
        # 获取当前批次的索引，确保最后一批样本不会超出数据集大小
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        
        # 返回当前批次的特征和标签，batch_indices作为张量索引，一维索引能够抓取二位张量的行，生成特定批次的张量数据
        yield features[batch_indices], labels[batch_indices]


#定义模型
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

#定义损失函数
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

#定义优化算法，实现随机梯度下降
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            #lr是学习率，也就是梯度更新的系数，由于这个梯度是所有batch_size的都加到一块儿了，所以除以batch_size
            param -= lr * param.grad / batch_size
            param.grad.zero_()

#从零开始的训练
def train_from_zero(plot=False):
    #设置真实参数值
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features:', features[:5],'\nlabel:', labels[0])

    if plot==True:
    # 绘图部分，将数据集的数据打印出来
        d2l.set_figsize()
        d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)

        plt.grid(True)
        plt.show()

    # batch_size = 10
    # for X, y in data_iter(batch_size, features, labels):
    #     print(X, '\n', y)
    #     break

    #初始化带优化参数，从正态分布中随机取样作为初始参数
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    #开始训练
    lr = 0.01
    num_epochs = 3
    net = linreg
    loss = squared_loss
    batch_size = 2

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'actual w:{true_w},actual b: {true_b}')
    print(f'caculate w: {w},b:{b}')

#----------------------------------#

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

#高级api实现
def train_from_torch():
    #设置标准数据
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    batch_size = 10

    #构建dataloader
    data_iter = load_array((features, labels), batch_size)
    #print(next(iter(data_iter)))

    #构建模型
    #Sequential类将多个层串联在一起，自动将上层输出转给下层输入
    #全连接层在Linear类中定义
    #第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1
    net = nn.Sequential(nn.Linear(2, 1))
    #选择网络中的第一个图层，设置权重和偏置
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    #定义损失函数，计算均方误差使用的是MSELoss类，默认情况下，它返回所有样本损失的平均值。
    loss = nn.MSELoss()

    #定义优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X) ,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            #print(f'grad:{net[0].weight.grad}')
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
        
    
    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)

    
def main():
    print('Start regression train')
    # train_from_zero()
    train_from_torch()

if __name__ == '__main__':
    main()