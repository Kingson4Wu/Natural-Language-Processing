# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)
C = create_co_matrix(corpus, vocab_size, window_size=1)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)

np.set_printoptions(precision=3)  # 有效位数为3位
print(C[0])
print(W[0])
print(U[0])

# plot
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()


# 基于 SVD 的降维

# 使用 NumPy 的 linalg 模块中的 svd 方法。linalg 是 linear algebra(线性代数)的简称。
# 创建一个共现矩阵，将其转化为 PPMI 矩阵，然后对其进行 SVD

# 对共现矩阵执行 SVD，并在图上绘制各个单词的二维向量(i 和 goodbye 重叠)
# 观察该图可以发现，goodbye 和 hello、you 和 i 位置接近，这是比较符 合我们的直觉的。但是，因为我们使用的语料库很小，有些结果就比较微 妙。
