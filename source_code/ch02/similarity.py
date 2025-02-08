# coding: utf-8
import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, cos_similarity


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]  #you的单词向量
c1 = C[word_to_id['i']]  #iの单词向量
print(cos_similarity(c0, c1))

# 求得单词向量间的相似度

# 余弦相似度直观地表示了“两个向量在多大程度上指向同一方向”。 两个向量完全指向相同的方向时，余弦相似度为 1;完全指向相反 的方向时，余弦相似度为 −1。
# 从上面的结果可知，you 和 i 的余弦相似度是 0.70 . . .。由于余弦相似度 的取值范围是 −1 到 1，所以可以说这个值是相对比较高的(存在相似性)