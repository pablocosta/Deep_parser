# coding=UTF-8

from Limpar_corpus import DependencyReader
from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.corpora import WikiCorpus
#corpus = DependencyReader()
#corpus.load("portuguese")


#arq_word = open("./corpus/arquivo_palavras.txt","r")
#arq_pos = open("./corpus/arquivo_pos.txt", "w+")
#arq_label = open("./corpus/arquivo_label.txt", "w+")
#bz2 = "./ptwiki-latest-pages-articles.xml.bz2"

#wiki = WikiCorpus(bz2, lemmatize=False, dictionary={})


#output = open("./wiki.txt", "w")
sentences = LineSentence("./corpus/arquivo_pos.txt")

#space = " "
#for text in wiki.get_texts():
    #output.write(space.join(str(x).replace("b'","") for x in text))
    #output.write("\n")
#output = open("./wiki.txt","w")
#a = arq_word.read()

#output.write(a)

#arquivo = open('./corpus/arquivo_palavras.txt')
#lines = [line for line in arq_word]


model = Word2Vec(sentences, min_count=1, workers=8, size=50) # an empty model, no training

model.save_word2vec_format(fname="./arquivo_pos.txt")
#model.save('./model_label')

