#Recebe como entrada um diretorio de corpus noo formato universal treebank, e transforma em 3 arquivos
#um contendo somente as sentencas (palavras)
#o segundo contendo somente PoS
#o terceiro somente labels de trnsicao


import sys
import os
from os import path
from nltk.corpus import DependencyCorpusReader
class Instance():
    '''
    Instance class
    '''
    def __init__(self):
        self.words = []
        self.pos = []
        self.heads = []


class DependencyReader():
    '''
    Dependency reader class
    '''
    def __init__(self):
        self.word_dict = {}
        self.pos_dict = {}
        self.train_instances = []
        self.test_instances = []

    def load(self, language):
        '''Loads training and test data for dependency parsing.'''
        self.word_dict = {}
        self.pos_dict = {}
        self.train_instances = []
        self.test_instances = []
        base_deppars_dir = path.join(path.dirname(__file__),"corpus")
        languages = ["danish","dutch","portuguese","english"]
        i = 0
        word_dict = {}
        pos_dict = {}
        feat_counts = {}
        if(language not in languages):
            print("Language does not exist: \"%s\": Available are: %s",(language,languages))
            return

        ### Create alphabet from training data
        n_sents = 0
        n_toks = 0
        word_id = 0
        pos_id = 0
        conll_file = open(path.join(base_deppars_dir+"/train/", language+"_train.conll"))

        self.word_dict["__START__"] = word_id # Start symbol
        word_id+=1
        self.word_dict["__STOP__"] = word_id # Stop symbol
        word_id+=1
        self.pos_dict["__START__"] = pos_id # Start symbol
        pos_id+=1
        self.pos_dict["__STOP__"] = pos_id # Stop symbol
        pos_id+=1

        for line in conll_file:
            line = line.rstrip()
            if len(line) == 0:
                n_sents+=1
                continue
            fields = line.split("\t")
            n_toks+=1
            word = fields[1]
            pos = fields[3]
            if word not in self.word_dict:
                self.word_dict[word] = word_id
                word_id+=1
            if pos not in self.pos_dict:
                self.pos_dict[pos] = pos_id
                pos_id+=1
        conll_file.close()

        print("Number of sentences: {0}".format(n_sents))
        print("Number of tokens: {0}".format(n_toks))
        print("Number of words: {0}".format(word_id))
        print("Number of pos: {0}".format(pos_id))


        ### Load training data
        self.train_instances = []
        inst = Instance()
        inst.words.append(self.word_dict["__START__"])
        inst.pos.append(self.pos_dict["__START__"])
        inst.heads.append(-1)
        conll_file = open(path.join(base_deppars_dir+"/train/", language+"_train.conll"))
        arq_word = open("./corpus/arquivo_palavras.txt", "w+")
        arq_pos = open("./corpus/arquivo_pos.txt", "w+")
        arq_label = open("./corpus/arquivo_label.txt", "w+")
        for line in conll_file:
            line = line.rstrip()
            if len(line) == 0:
                n_sents+=1
                self.train_instances.append(inst)
                inst = Instance()
                inst.words.append(self.word_dict["__START__"])
                inst.pos.append(self.pos_dict["__START__"])
                inst.heads.append(-1)
                continue
            fields = line.split("\t")

            word = fields[1]
            pos = fields[3]
            head = int(fields[6])
            label = fields[7]

            if word not in self.word_dict:
                word_id = -1
            else:
                word_id = self.word_dict[word]
            if pos not in self.pos_dict:
                pos_id = -1
            else:
                pos_id = self.pos_dict[pos]

            inst.words.append(word_id)
            inst.pos.append(pos_id)
            inst.heads.append(head)
            if fields[0] == "1":
                arq_word.write("\n"+word)
                arq_pos.write("\n"+pos)
                arq_label.write("\n"+label)
            else:

                arq_word.write(" "+word)
                arq_pos.write(" "+pos)
                arq_label.write(" "+label)





        conll_file.close()
        arq_label.close()
        arq_pos.close()


        ### Load test data
        self.test_instances = []
        inst = Instance()
        inst.words.append(self.word_dict["__START__"])
        inst.pos.append(self.pos_dict["__START__"])
        inst.heads.append(-1)
        conll_file = open(path.join(base_deppars_dir+"/test/", language+"_test.conll"))
        for line in conll_file:
            line = line.rstrip()
            if len(line) == 0:
                n_sents+=1
                self.test_instances.append(inst)
                inst = Instance()
                inst.words.append(self.word_dict["__START__"])
                inst.pos.append(self.pos_dict["__START__"])
                inst.heads.append(-1)
                continue
            fields = line.split("\t")

            word = fields[1]
            pos = fields[3]
            head = int(fields[6])

            if word not in self.word_dict:
                word_id = -1
            else:
                word_id = self.word_dict[word]
            if pos not in self.pos_dict:
                pos_id = -1
            else:
                pos_id = self.pos_dict[pos]

            if fields[0] == "1":
                arq_word.write("\n"+word)
            else:
                arq_word.write(" "+word)

            inst.words.append(word_id)
            inst.pos.append(pos_id)
            inst.heads.append(head) # gold heads

        conll_file.close()
        arq_word.close()
    def load_senteces_train(self, languages):
        frase = dict()
        base_deppars_dir = path.join(path.dirname(__file__),"corpus/train/")
        for language in languages:
            arquivo = open(path.join(base_deppars_dir, language+"_train.conll"))
            i = 0
            ok = 1
            frase[0] = []
            for line in arquivo:
                frase[i].append(line)
                line = line.rstrip()
                if line == "":
                    i +=1
                    frase[i] = []


            arquivo.close()

        return frase



