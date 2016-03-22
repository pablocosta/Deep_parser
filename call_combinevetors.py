import gensim

space = " "
inp = "ptwiki-latest-pages-articles.xml.bz2"
wiki = gensim.corpora.WikiCorpus(inp, lemmatize=False, dictionary={})
def gettext():
    for i, text in enumerate(wiki.get_texts()):
        yield text

model = gensim.models.Word2Vec(size=50) # an empty model, no training
model.build_vocab(gettext())
model.train(gettext())
model.save_word2vec_format(fname="./teste.txt")
