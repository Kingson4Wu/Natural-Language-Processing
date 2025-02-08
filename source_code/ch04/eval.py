# coding: utf-8
import sys
sys.path.append('..')
from common.util import most_similar, analogy
import pickle


pkl_file = 'cbow_params.pkl'
# pkl_file = 'skipgram_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

# most similar task
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

# analogy task
print('-'*50)
analogy('king', 'man', 'queen',  word_to_id, id_to_word, word_vecs)
analogy('take', 'took', 'go',  word_to_id, id_to_word, word_vecs)
analogy('car', 'cars', 'child',  word_to_id, id_to_word, word_vecs)
analogy('good', 'better', 'bad',  word_to_id, id_to_word, word_vecs)


#  CBOW 模型的评价

# 我们看一下结果。首先，在查询 you 的情况下，近似单词中出现了人 称代词 i(= I)和 we 等。接着，查询 year，可以看到 month、week 等表示时间区间的具有相同性质的单词。然后，查询 toyota，可以得到 ford、 mazda 和 nissan 等表示汽车制造商的词汇。从这些结果可以看出，由 CBOW 模型获得的单词的分布式表示具有良好的性质。
# 此外，由 word2vec 获得的单词的分布式表示不仅可以将近似单词聚 拢在一起，还可以捕获更复杂的模式，其中一个具有代表性的例子是因 “king − man + woman = queen”而出名的类推问题(类比问题)。更准确 地说，使用 word2vec 的单词的分布式表示，可以通过向量的加减法来解决类推问题。

# 这里的类推问题的结果看上去非常好。不过遗憾的是，这是笔者 特意选出来的能够被顺利解决的问题。实际上，很多问题都无法 获得预期的结果。这是因为 PTB 数据集的规模还是比较小。如 果使用更大规模的语料库，可以获得更准确、更可靠的单词的分 布式表示，从而大大提高类推问题的准确率。