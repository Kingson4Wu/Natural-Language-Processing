# coding: utf-8
import sys
sys.path.append('..')  # 为了引入父目录的文件而进行的设定
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot


window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])


# CBOW 模型的学习和一般的神经网络的学习完全相同。首先，给神 经网络准备好学习数据。然后，求梯度，并逐步更新权重参数。

# 这里使用的小型语料库并没有给出很好的结果。当 然，主要原因是语料库太小了。如果换成更大、更实用的语料库，相信会获 得更好的结果。但是，这样在处理速度方面又会出现新的问题，这是因为当 前这个 CBOW 模型的实现在处理效率方面存在几个问题。下一章我们将改 进这个简单的 CBOW 模型，实现一个“真正的”CBOW 模型。
