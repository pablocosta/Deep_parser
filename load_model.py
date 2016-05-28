import keras
import sys
import numpy as np
from os import path
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.core import Dense, Activation, Dropout, Merge, Reshape
from keras.layers.embeddings import Embedding
from nltk.parse import DependencyGraph, DependencyEvaluator
from parser import TransitionParser
import pickle

class Model():



    def __init__(self, model, test_data, caminho, parser):
        "./corpus/test/portuguese_test.conll"
        self.test_data = test_data
        self.predicted = []
        self.parser = parser
        if model is not None:
            self.model = model

        else:

            self.caminho = caminho

            self.words = self.Load_embedding_file("word.txt")
            self.tags = self.Load_embedding_file("pos.txt")
            self.labels = self.Load_embedding_file("label.txt")

            self.output_dim = 77


            word_features = Sequential()
            word_features.add(Embedding(input_dim=len(self.words.keys())+1,input_length=18, output_dim=50,  mask_zero=True))

            pos_features = Sequential()
            pos_features.add(Embedding(input_dim=len(self.tags.keys())+1,input_length=18, output_dim=50, mask_zero=True))

            label_features = Sequential()
            label_features.add(Embedding(input_dim=len(self.labels.keys())+1,input_length=12, output_dim=50,  mask_zero=True))

            self.model = Sequential()

            self.model.add(Merge([word_features, pos_features, label_features], mode='concat', concat_axis = 1))
            self.model.add(Reshape((48*50,)))
            self.model.add(Dense(output_dim=200, W_regularizer=l2(0.01)))
            self.model.add(Dropout(0.5))

            # To-do: Modelar a ativacao tripla X 3
            self.model.add(Activation("tanh"))
            self.model.add(Dense(output_dim=self.output_dim, input_dim=200))
            self.model.add(Activation('softmax'))
            adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)
            self.model.compile(loss='categorical_crossentropy', optimizer=adagrad)
            parser_std = TransitionParser('arc-standard')
            self.parser = self.extract_training_data_arc_standard(parser_std)

    def create_dict_inx(self, dictionary):
        return_dict = dict()
        i = 0
        for key in dictionary.keys():
            return_dict[key] = i
            i += 1
        return return_dict
    def evaluate_parser(self):
        arquivo = open(self.test_data)
        a = arquivo.read()
        gold_sent = [DependencyGraph(entry) for entry in a.split('\n\n') if entry]
        de = DependencyEvaluator(self.predicted, gold_sent)
        print(de.eval())


    def id_tensor_to_one_hot_tensor(self, tensor_2d, one_hot_dim=None):
        """
        :param tensor_2d: numpy array of sequences of ids
        :param one_hot_dim: if not specified, max value in tensor_2d + 1 is used
        :return:
        """

        # return np_utils.to_categorical(tensor_2d)
        if not one_hot_dim:
            one_hot_dim = tensor_2d.max() + 1

        tensor_3d = np.zeros((tensor_2d.shape[0], tensor_2d.shape[1], one_hot_dim), dtype=np.bool8)
        for (i, j), val in np.ndenumerate(tensor_2d):
            tensor_3d[i, j, val] = 1
        return tensor_3d

    def to_one_hot(self, arq):
        text_ = dict()
        i = 0
        for line in arq:
            text_[line.rstrip()] = i
            i = i + 1
        keys2 = [key for key in text_.values()]
        keys2.sort()
        keys = np.array(keys2, ndmin=2)
        arq.close()
        return (text_, self.id_tensor_to_one_hot_tensor(keys, one_hot_dim=len(text_.keys())))


    def return_weights(self, dict_vec, dict_indx):
        weights = np.zeros((len(dict_vec.keys()) + 1, 50))
        for key in dict_indx.keys():
            weights[dict_indx[key], :] = dict_vec[key]
        return weights

    def Load_embedding_file(self, file_embedding_model):
        arquivo = open(path.join("./embeddings/", file_embedding_model))
        embeddings = dict()

        for s in arquivo:
            s = s.split(" ")
            if s[0] not in embeddings.keys():
                embeddings[s[0]] = [float(x) for x in s[1:]]
        arquivo.close()
        return embeddings

    def create_dict_inx(self, dictionary):
        return_dict = dict()
        i = 0
        for key in dictionary.keys():
            return_dict[key] = i
            i += 1
        return return_dict
    def load_dict(self):
        a = open("./corpus/temp/dict_onehot.pickl", 'rb')
        object = pickle.load(a)

        return object

    def parse(self, parser_std, model, path_test):
        # carrega gold
        arquivo = open(path_test)
        a = arquivo.read()
        graphs = [DependencyGraph(entry) for entry in a.split('\n\n') if entry]
        words = self.Load_embedding_file("word.txt")
        tags = self.Load_embedding_file("pos.txt")
        labels = self.Load_embedding_file("label.txt")
        dict_op = self.load_dict()
        words_indx = self.create_dict_inx(words)
        tags_indx = self.create_dict_inx(tags)
        labels_indx = self.create_dict_inx(labels)

        set_parser = parser_std.parse(graphs, model, words_indx, tags_indx, labels_indx, dict_op)

        return set_parser

    def load_weights(self):
        self.model.load_weights(self.caminho)

    def extract_training_data_arc_standard(self, parser_std):
        arquivo = open("./corpus/train/portuguese_train.conll", "r")
        a = arquivo.read()
        graphs = [DependencyGraph(entry) for entry in a.split('\n\n') if entry]
        input_file = open("./corpus/temp/parc_test.pickl", "wb")
        parser_std._create_training_examples_arc_std(graphs, input_file)
        arquivo.close()
        input_file.close()
        return parser_std

    def parse_language(self):

        self.predicted = self.parse(self.parser, self.model, self.test_data)
