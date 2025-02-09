# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


N = 2  # mini-batch的大小
H = 3  # 隐藏状态向量的维数
T = 20  # 时序数据的长度

dh = np.ones((N, H))

np.random.seed(3)

Wh = np.random.randn(H, H) # 梯度爆炸(exploding gradients)
#Wh = np.random.randn(H, H) * 0.5 # 梯度消失 (vanishing gradients)

norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

print(norm_list)

# 绘制图形
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel('time step')
plt.ylabel('norm')
plt.show()


# 反向传播时梯度的值通过 MatMul 节点时会如何变化呢?
# 观察梯度大小的变化
# 梯度 dh 的大小随时间步长呈指数级增加
# 可知梯度的大小随时间步长呈指数级增加，这就是梯度 爆炸(exploding gradients)。如果发生梯度爆炸，最终就会导致溢出，出 现 NaN(Not a Number，非数值)之类的
