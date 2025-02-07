# coding: utf-8
import sys
sys.path.append('..')  # 为了引入父目录的文件而进行的设定
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 初始化权重和偏置
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        # 生成层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        # 将所有的权重和偏置整理到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


# 神经网络的实现

# 实现一个具有一个隐藏层的神经网络

# 初始化程序接收 3 个参数。input_size 是输入层的神经元数，hidden_ size 是隐藏层的神经元数，output_size 是输出层的神经元数。在内部实 现中，首先用零向量(np.zeros())初始化偏置，再用小的随机数(0.01 * np.random.randn())初始化权重。通过将权重设成小的随机数，学习可以更 容易地进行。接着，生成必要的层，并将它们整理到实例变量 layers 列表 中。最后，将这个模型使用到的参数和梯度归纳在一起。
# 因为Softmax with Loss层和其他层的处理方式不同，所以不将 它放入 layers 列表中，而是单独存储在实例变量 loss_layer 中。
# 接着，我们为 TwoLayerNet 实现 3 个方法，即进行推理的 predict() 方法、正向传播的 forward() 方法和反向传播的 backward() 方法
# 这个实现非常清楚。因为我们已经将神经网络中要用的处理 模块实现为了层，所以这里只需要以合理的顺序调用这些层的 forward() 方 法和 backward() 方法即可。


