# coding: utf-8
import sys
sys.path.append('..')  # 为了引入父目录的文件而进行的设定
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet


# 设定超参数
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

# 学习用的变量
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    # 打乱数据
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        # 计算梯度，更新参数
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 定期输出学习过程
        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d |  iter %d / %d | loss %.2f'
                  % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0


# 绘制学习结果
plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('iterations (x10)')
plt.ylabel('loss')
plt.show()

# 绘制决策边界
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# 绘制数据点
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()

# 学习用的代码 
# 首先，读入学习数据，生成神经网 络(模型)和优化器。然后，按照之前介绍的学习的 4 个步骤进行学习。另 外，在机器学习领域，通常将针对具体问题设计的方法(神经网络、SVM 等)称为模型。
# 首先，在代码❶的地方设定超参数。具体而言，就是设定学习的 epoch 数、mini-batch 的大小、隐藏层的神经元数和学习率。接着，在代码❷的地 方进行数据的读入，生成神经网络(模型)和优化器。我们已经将 2 层神经 网络实现为了 TwoLayerNet 类，将优化器实现为了 SGD 类，这里直接使用它 们就可以。
# epoch 表示学习的单位。1 个 epoch 相当于模型“看过”一遍所有的 学习数据(遍历数据集)。这里我们进行 300 个 epoch 的学习。
# 在进行学习时，需要随机选择数据作为 mini-batch。这里，我们以epoch 为单位打乱数据，对于打乱后的数据，按顺序从头开始抽取数据。数 据的打乱(准确地说，是数据索引的打乱)使用 np.random.permutation() 方 法。给定参数 N，该方法可以返回 0 到 N − 1 的随机序列,调用 np.random.permutation() 可以随机打乱数据的索引。
# 接着，在代码 ❹ 的地方计算梯度，更新参数。最后，在代码 ❺ 的地方定 期地输出学习结果。这里，每 10 次迭代计算 1 次平均损失，并将其添加到变量 loss_list 中。以上就是学习用的代码。

# 这里实现的神经网络的学习用的代码在本书其他地方也可以使用。 因此，本书将这部分代码作为 Trainer 类提供出来。使用 Trainer 类，可以将神经网络的学习细节嵌入 Trainer类。详细的用法将 在 1.4.4 节说明。

# 运行一下上面的代码(ch01/train_custom_loop.py)就会发现，向终端 输出的损失的值在平稳下降。

# 随着学习的进行，损失在减小。我们的神经网络正在 朝着正确的方向学习!接下来，我们将学习后的神经网络的区域划分(也称 为决策边界)可视化

# 学习后的神经网络可以正确地捕获“旋涡”这个模式。 也就说，模型正确地学习了非线性的区域划分。像这样，神经网络通过隐藏 层可以实现复杂的表现力。深度学习的特征之一就是叠加的层越多，表现力 越丰富。

# 损失的图形:横轴是学习的迭代次数(刻度值的 10 倍)，竖轴是每 10 次迭代的平 均损失
# 学习后的神经网络的决策边界(用不同颜色描绘神经网络识别的各个类别的区域)

