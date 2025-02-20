# coding: utf-8
import sys
sys.path.append('..')  # 为了引入父目录的文件而进行的设定
from common.optimizer import SGD
from common.trainer import Trainer
from dataset import spiral
from two_layer_net import TwoLayerNet


# 设定超参数
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
trainer.plot()

# 这个类的初始化程序接收神 经网络(模型)和优化器
# Trainer 类有 plot() 方法，它将 fit() 方法记录的损失(准确地 说，是按照 eval_interval 评价的平均损失)在图上画出来
# 将之前展 示的学习用的代码交给 Trainer 类负责，代码变简洁了。

