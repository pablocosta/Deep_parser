# coding=UTF-8

from gensim.models.word2vec import LineSentence, Word2Vec

sentences = LineSentence("./embeddings/arquivo_label.txt")



model = Word2Vec(sentences, min_count=1, workers=16, size=50) # an empty model, no training

model.save_word2vec_format(fname="./embeddings/labels.txt")
