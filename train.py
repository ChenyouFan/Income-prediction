import numpy as np
import math


def logistic_regression(training_data, label, weights, bias):
    """Logistic回归

        二元分类，sigmoid激活，输出预测结果

    Args:
        training_data: 输入的数据
        label: 真实结果
        weights: 权重
        bias: 偏置

    Returns:
        prediction: 预测结果
        loss_value: 损失函数值
    """
    z = np.dot(training_data, weights) + bias
    prediction = np.clip(1 / (1 + np.exp(-z)), 10 ** -6, 1 - 10 ** -6)      # 用clip限制最值，防止overflow
    # 交叉熵损失函数
    loss_value = (-(np.dot(label.T, np.log(prediction)) + np.dot((1 - label).T, np.log(1 - prediction)))) / training_data.shape[0]

    return prediction, loss_value


# 读取数据
with open('./X_train') as f:
    next(f)
    x = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
with open('./Y_train') as f:
    next(f)
    y = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)[:, np.newaxis]

# 标准化
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x = (x - x_mean) / (x_std + 10**-10)        # 防止分母为0

# 打乱数据集
np.random.seed(116)
np.random.shuffle(x)
np.random.seed(116)
np.random.shuffle(y)

# 划分训练集和验证集
x_train = x[:math.floor(len(x)*0.8), :]
y_train = y[:math.floor(len(y)*0.8), :]
x_val = x[math.floor(len(x)*0.8):, :]
y_val = y[math.floor(len(y)*0.8):, :]

# 随机初始化
np.random.seed(17)
w = np.random.normal(size=[510, 1])
b = np.random.normal(size=1)

iterations = 5500       # 迭代次数
batch_size = 2048
beta = 0.95      # SGDM超参数
# 动量初始化
vw = np.zeros((510, 1))
vb = 0
index = np.arange(len(x_train))

for i in range(iterations):

    for j in range(math.floor(len(x_train) / batch_size)):
        x_batch_train = x_train[j * batch_size: (j + 1) * batch_size, :]
        y_batch_train = y_train[j * batch_size: (j + 1) * batch_size, :]

        # 前向传播预测结果
        A, loss = logistic_regression(x_batch_train, y_batch_train, w, b)

        # 反向传播，链式法则求导，计算梯度
        dz = A - y_batch_train        # dz=dL/dA * dA/dL  L为损失函数，A=sigmoid(z)
        dw = np.dot(x_batch_train.T, dz) / x_batch_train.shape[0]
        db = np.sum(dz, axis=0) / x_batch_train.shape[0]

        # SGDM优化
        vw = beta * vw + (1 - beta) * dw
        vb = beta * vb + (1 - beta) * db
        w -= vw
        b -= vb

    if i % 100 == 0:  # 每100轮用验证集验证一次
        A_val, loss_val = logistic_regression(x_val, y_val, w, b)
        A_train, loss_train = logistic_regression(x_train, y_train, w, b)
        print('after ' + str(i) + ' epoch, the validation loss is:', float(loss_val), ',the training loss is:', float(loss_train))

np.save('weights.npy', w)
np.save('bias.npy', b)
