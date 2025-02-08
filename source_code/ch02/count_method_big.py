# coding: utf-8

# pip3 install scikit-learn

import sys
sys.path.append('..')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb


window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting  co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('calculating PPMI ...')
W = ppmi(C, verbose=True)

print('calculating SVD ...')
try:
    # truncated SVD (fast!)
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5,
                             random_state=None)
except ImportError:
    # SVD (slow)
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)


# 为了执行 SVD，我们使用了 sklearn 的 randomized_svd() 方法。 该方法通过使用了随机数的 Truncated SVD，仅对奇异值较大的部分进行 计算，计算速度比常规的 SVD 快。剩
# 因为使用了随机数，所以在使用 Truncated SVD 的情况下，每次的结果都不一样

# 使用语料库，计算上下文中的单词数量，将它们转化 PPMI 矩阵，再基于 SVD 降维 获得好的单词向量。这就是单词的分布式表示，每个单词表示为固定长度的 密集向量。

