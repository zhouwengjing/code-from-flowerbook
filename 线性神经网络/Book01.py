import random
import torch
import matplotlib
import matplotlib.pyplot as plt
# from d2l import torch as d2l

matplotlib.use("TkAgg")

# 通过rc参数修改字体为黑体，就可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # STHeiTi
# 通过rc参数修改字符显示，就可以正常显示符号
plt.rcParams['axes.unicode_minus'] = False


# 生成数据集


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])

# 比较三个不同的散点图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
ax1.scatter(features[:, 0].detach().numpy(), labels.detach().numpy(), alpha=0.5)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Labels')
ax1.set_title('Feature1 vs Labels\n(有明显的线性关系)')
ax2.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), alpha=0.5)
ax2.set_xlabel('Feature 2')
ax2.set_ylabel('Labels')
ax2.set_title('Feature2 vs Labels\n(系数为2的线性关系)')
ax3.scatter(features[:, 0].detach().numpy(), features[:, 1].detach().numpy(), alpha=0.5)
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.set_title('Feature1 vs Feature2\n(随机分布)')
plt.tight_layout()


# plt.show()


def data_iter(batch_size, features, labels):
    """数据迭代器，用于按批次获取数据"""
    num_examples = len(features)  # 获取样本总数
    indices = list(range(num_examples))  # 创建索引列表 [0, 1, 2, ..., num_examples-1]

    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)  # 打乱索引顺序，实现随机采样

    # 循环遍历所有样本，步长为batch_size
    for i in range(0, num_examples, batch_size):
        # 获取当前批次的索引，确保不超出范围
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])

        # 使用yield返回当前批次的数据（特征和标签）
        # yield使得这个函数成为一个生成器，可以迭代使用
        yield features[batch_indices], labels[batch_indices]


# 测试数据迭代器
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break  # 只查看第一个批次

# 初始化模型参数
# w: 权重参数，从正态分布中随机初始化，形状为(2, 1)，需要计算梯度
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)

# b: 偏置参数，初始化为0，需要计算梯度
b = torch.zeros(1, requires_grad=True)


def linreg(X, w, b):
    """线性回归模型
    参数:
        X: 输入特征矩阵，形状为(batch_size, num_features)
        w: 权重参数，形状为(num_features, 1)
        b: 偏置参数，形状为(1,)
    返回:
        预测值 y_hat = Xw + b
    """
    return torch.matmul(X, w) + b  # 矩阵乘法 X*w 然后加上偏置 b


def squared_loss(y_hat, y):
    """均方损失函数
    参数:
        y_hat: 模型预测值
        y: 真实标签值
    返回:
        每个样本的损失值，形状与y_hat相同
    """
    # 将y的形状调整为与y_hat相同（确保维度一致）
    # 计算平方损失：(预测值-真实值)^2 / 2
    # 除以2是为了后续求导时系数为1，简化计算
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降优化器
    参数:
        params: 需要优化的参数列表 [w, b]
        lr: 学习率 (learning rate)
        batch_size: 批次大小，用于归一化梯度
    """
    # 使用torch.no_grad()上下文管理器，避免跟踪梯度更新操作
    with torch.no_grad():
        for param in params:
            # 更新参数: param = param - lr * gradient / batch_size
            # 除以batch_size是对梯度进行平均，使得学习率与批次大小无关
            param -= lr * param.grad / batch_size

            # 清空当前参数的梯度，为下一次计算做准备
            param.grad.zero_()


# ========== 训练过程 ==========

# 超参数设置
lr = 0.03  # 学习率
num_epochs = 3  # 训练轮数

# 定义模型和损失函数（这里只是起别名，便于理解）
net = linreg  # 网络模型
loss = squared_loss  # 损失函数

# 开始训练循环
for epoch in range(num_epochs):
    # 遍历每个小批量数据
    for X, y in data_iter(batch_size, features, labels):
        # 前向传播：计算当前批次的预测值
        y_hat = net(X, w, b)

        # 计算损失：l的形状是(batch_size, 1)
        l = loss(y_hat, y)

        # 反向传播：计算梯度
        # 先对批次内所有样本的损失求和，然后反向传播
        l.sum().backward()

        # 使用随机梯度下降更新参数
        sgd([w, b], lr, batch_size)

    # 每个epoch结束后，评估在整个训练集上的损失
    with torch.no_grad():  # 不需要计算梯度，节省内存
        train_l = loss(net(features, w, b), labels)  # 计算所有样本的损失
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')  # 打印平均损失

# 训练结束后，评估参数估计的准确性
print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b - b}')