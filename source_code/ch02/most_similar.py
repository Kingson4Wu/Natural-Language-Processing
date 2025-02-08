# coding: utf-8
import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, most_similar


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)

# 将 you 作为查询词， 显示与其相似的单词

# 这个结果只按降序显示了 you 这个查询词的前 5 个相似单词，各个单 词旁边的值是余弦相似度。观察上面的结果可知，和 you 最接近的单词有 3 个，分别是 goodbye、i(= I)和 hello。因为 i 和 you 都是人称代词，所以 二者相似可以理解。但是，goodbye 和 hello 的余弦相似度也很高，这和我们的感觉存在很大的差异。一个可能的原因是，这里的语料库太小了。后面我们会用更大的语料库进行相同的实验。

