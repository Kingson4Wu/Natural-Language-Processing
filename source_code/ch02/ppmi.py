# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)  # 有效位数为3位
print('covariance matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)

# 将共现矩阵转化为 PPMI 矩阵
# 将共现矩阵转化为了 PPMI 矩阵。此时，PPMI 矩 阵的各个元素均为大于等于 0 的实数。我们得到了一个由更好的指标形成的 矩阵，这相当于获取了一个更好的单词向量。
