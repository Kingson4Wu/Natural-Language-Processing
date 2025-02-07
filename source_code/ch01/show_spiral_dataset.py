# coding: utf-8
import sys
sys.path.append('..')  # 为了引入父目录的文件而进行的设定
from dataset import spiral
import matplotlib.pyplot as plt


x, t = spiral.load_data()
print('x', x.shape)  # (300, 2)
print('t', t.shape)  # (300, 3)

# 绘制数据点
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()

# 螺旋状数据集
# 本书在 dataset 目录中提供了几个便于处理数据集的类，本节将使用其 中的 dataset/spiral.py 文件。这个文件中实现了读取螺旋(旋涡)状数据 的类
# 要从 ch01 目录的 dataset 目录引入 spiral.py。因此， 上面的代码通过 sys.path.append('..') 将父目录添加到了 import 的检索路 径中。
# 然后，使用 spiral.load_data() 进行数据的读入。此时，x 是输入数据， t 是监督标签。观察 x 和 t 的形状，可知它们各自有 300 笔样本数据，其中 x 是二维数据，t 是三维数据。另外，t 是 one-hot 向量，对应的正确解标签 的类标记为 1，其余的标记为 0。下面，我们把这些数据绘制在图上
# 学习用的螺旋状数据集(用 × ▲●分别表示 3 个类)
# 输入是二维数据，类别数是 3。观察这个数据集可知， 它不能被直线分割。因此，我们需要学习非线性的分割线。那么，我们的神 经网络(具有使用非线性的 sigmoid 激活函数的隐藏层的神经网络)能否正 确学习这种非线性模式呢?

# 因为这个实验相对简单，所以我们不把数据集分成训练数据、验 证数据和测试数据。不过，实际任务中会将数据集分为训练数据 和测试数据(以及验证数据)来进行学习和评估。

