import keras
import numpy as np
import numpy.ma as ma
import os
import copy
import pickle
from os import path
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.core import Dense, Activation, Dropout, Merge, Reshape
from keras.layers.embeddings import Embedding
from nltk.parse import DependencyGraph
from parser import TransitionParser


def create_dict_inx(dictionary):
    return_dict = dict()
    i = 0
    for key in dictionary.keys():
        return_dict[key] = i
        i += 1
    return return_dict


def return_weights(dict_vec, dict_indx):
    weights = np.zeros((len(dict_vec.keys()) + 1, 50))
    for key in dict_indx.keys():
        weights[dict_indx[key], :] = dict_vec[key]
    return weights


def pre_trainneural(parser_std):
    # Carregando embeddings
    # w2v_word = Word2Vec.load_from_w2v_file()
    # w2v_pos = Word2Vec.load_from_w2v_file()
    # w2v_label = Word2Vec.load_from_w2v_file()
    # Para metros do treinameto
    batch_size = 128
    nb_epoch = 1

    """ TO-DO
        fazer funcao X^3
    """

    words = Load_embedding_file("word.txt")
    tags = Load_embedding_file("pos.txt")
    labels = Load_embedding_file("label.txt")
    # make_softmax_arq(labels)

    dirs = [os.listdir("./corpus/test"), os.listdir("./corpus/train")]

    for file in dirs:
        if "test" in file[0]:
            clean_corpus("./corpus/test/" + file[0], parser_std)
        else:
            clean_corpus("./corpus/train/" + file[0], parser_std)

    output_dim = 77

    # create dictionary for the index
    words_indx = create_dict_inx(words)
    tags_indx = create_dict_inx(tags)
    labels_indx = create_dict_inx(labels)

    w_weights = return_weights(words, words_indx)
    p_weights = return_weights(tags, tags_indx)
    l_weights = return_weights(labels, labels_indx)


    word_features = Sequential()
    word_features.add(Embedding(input_dim=len(words.keys())+1, output_dim=50, input_shape=(len(words.keys()),),  mask_zero=True, weights=[w_weights]))

    pos_features = Sequential()
    pos_features.add(Embedding(input_dim=len(tags.keys())+1, output_dim=50, input_shape=(len(tags.keys()),), mask_zero=True, weights=[p_weights]))

    label_features = Sequential()
    label_features.add(Embedding(input_dim=len(labels.keys())+1, output_dim=50, input_shape=(len(labels.keys()),),  mask_zero=True, weights=[l_weights]))

    model = Sequential()
    model.add(Merge([word_features, pos_features, label_features], mode='concat'))
    model.add(Reshape((len(labels.keys())+ len(tags.keys()) + len(words.keys()),)))
    model.add(Dense(output_dim=200, input_dim=50 * 48, W_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    # To-do: Modelar a ativacao tripla X 3
    model.add(Activation("tanh"))
    model.add(Dense(output_dim=output_dim, input_dim=200))
    model.add(Activation('softmax'))
    adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad)
    # Extract one-hot repersentations

    (dict_op, one_hot) = to_one_hot(open("./corpus/temp/softmax_arq.txt"))
    dict_opaux = copy.deepcopy(dict_op)
    for element in range(0, len(one_hot[0])):
        for key in dict_op.keys():
            if dict_opaux[key] == element:
                dict_op[key] = one_hot[0][element]
    #extract_training_data_arc_standard()
    X, Y = get_all(words_indx, tags_indx, labels_indx, dict_op)
    X = np.array(X)
    Y = np.array(Y)
    model.fit(X=X, y=Y, nb_epoch=nb_epoch, batch_size=batch_size)

    return model


def Load_embedding_file(file_embedding_model):
    arquivo = open(path.join("./embeddings/", file_embedding_model))
    embeddings = dict()

    for s in arquivo:
        s = s.split(" ")
        if s[0] not in embeddings.keys():
            embeddings[s[0]] = [float(x) for x in s[1:]]
    arquivo.close()
    return embeddings

def Get_features_fromop(list_fatures, dict_op):

    return np.array(dict_op[list_fatures])

def Get_features_(list, dict):

    return [dict[l] for l in list]



def get_all(words, tags, labels, dict_op):
    # this function get all configurations from the training files
    i = 0
    aux = 0
    X_ = []
    Y_ = []

    # Todo o conjunto de treinamento esta no arquivo input_file da forma Y [features_word] [features_pos] [features_label]
    #input_file = open(, "rb")

    a = open("./corpus/temp/parc_test.pickl", 'rb')
    object = pickle.load(a)
    for element in object:
        x = []
        for l in element:

            if i == 0:
                Y_.append(Get_features_fromop(l, dict_op))
                i += 1
            else:
                if i == 1:
                    x.append(Get_features_(l, words))
                if i == 2:
                    x.append(Get_features_(l, tags))
                if i == 3:
                    x.append(Get_features_(l, labels))
                i += 1
        i = 0
        X_.append(x)
    #for chaves in (dict_train[j]).keys():
        #fields = chaves.split(" ")
        #word1_embed = words[fields[0]]
        #word2_embed = words[fields[1]]
        #tag_embed = tags[fields[2]]
        #label_embed = labels[fields[3]]

        #X.append(ma.concatenate([word1_embed, word2_embed, tag_embed, label_embed]))
        #Y.append(np.array(dict_op[dict_train[j][fields[0] + " " + fields[1] + " " + fields[2] + " " + fields[3]]]))


        # for j in range(0,len(senteces)):
        # for line in senteces[i]:
        # fields = line.split("\t")
        # if len(fields) == 1:
        # i += 1
        # e = input()
        # else:
        # training lexicalized
        # O parametro Y do metodo train_on_batch deve ser o que... a palavra alvo, o arclabel...
        # https://github.com/fchollet/keras/blob/master/examples/skipgram_word_embeddings.py
        # https://github.com/fchollet/keras/blob/master/tests/auto/test_sequential_model.py
        # artigo do chines ir.hit.edu.cn/~jguo/papers/acl2015-clnndep.pdf
        # artigo da chen http://cs.stanford.edu/~danqi/papers/emnlp2014.pdf
        # word1_embed= words[fields[1].lower()]
        # senteces[language] get line witch arc is mento

        # lines= senteces[i]
        # fields2 = lines[int(fields[6])-1].split("\t")
        # if int(fields[6]) == 0:
        # word2_embed=labels[fields[7]]
        # else:
        # word2_embed=words[fields2[1].lower()]
        # tag_embed=tags[fields[3]]
        # label_embed=labels[fields[7]]
        # X.append(ma.concatenate([word1_embed,word2_embed,tag_embed,label_embed]))
        # print(line)
        # print(fields[0]+":"+fields[6])
        # print(dict_train[i][fields[0]+":"+fields[6]])
        # Y.append(np.array(dict_op[dict_train[i][fields[0]+":"+fields[6]]+"\n"]))

    return X_, Y_


def to_one_hot(arq):
    text_ = dict()
    i = 0
    for line in arq:
        text_[line.rstrip()] = i
        i = i + 1
    keys2 = [key for key in text_.values()]
    keys2.sort()
    keys = np.array(keys2, ndmin=2)
    arq.close()
    return (text_, id_tensor_to_one_hot_tensor(keys, one_hot_dim=len(text_.keys())))


def make_softmax_arq(labels):
    operations_arq = open("./corpus/temp/softmax_arq.txt", "w")
    op = 0
    for i in range(0, 2):
        for l in labels.keys():
            if op == 0:
                operations_arq.write("LEFTARC" + ":" + l + "\n")
            else:
                operations_arq.write("RIGHTARC" + ":" + l + "\n")
        op = 1
    operations_arq.write("SHIFT")


def id_tensor_to_one_hot_tensor(tensor_2d, one_hot_dim=None):
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


def extract_training_data_arc_standard():
    arquivo = open("./corpus/train/portuguese_train.conll", "r")
    a = arquivo.read()
    graphs = [DependencyGraph(entry) for entry in a.split('\n\n') if entry]
    input_file = open("./corpus/temp/parc_test.pickl", "wb")
    parser_std = TransitionParser('arc-standard')
    parser_std._create_training_examples_arc_std(graphs, input_file)
    arquivo.close()
    input_file.close()
    #input_file.close()


def clean_corpus(path, parser_std):
    arquivo = open(path)
    a = arquivo.read()
    new_training_data = open(path + "1", "w")
    graphs = [DependencyGraph(entry) for entry in a.split('\n\n') if entry]
    for depgraph in graphs:
        if parser_std._is_projective(depgraph):
            new_training_data.write(depgraph.to_conll(style=10) + "\n")

    os.remove(arquivo.name)
    new_training_data.close()
    os.rename(new_training_data.name, (new_training_data.name).replace("1", ""))
    new_training_data.close()


def parse(parser_std, model, path_test):
    # carrega gold
    arquivo = open(path_test)
    a = arquivo.read()
    graphs = [DependencyGraph(entry) for entry in a.split('\n\n') if entry]
    words = Load_embedding_file("word.txt")
    tags = Load_embedding_file("pos.txt")
    labels = Load_embedding_file("label.txt")
    (dict_op, one_hot) = to_one_hot(open("./corpus/temp/softmax_arq.txt"))

    set_parser = parser_std.parse(graphs, model, words, tags, labels, dict_op)

    return set_parser


# carrega analisador
parser_std = TransitionParser('arc-standard')

# Treinamento do modelo
model = pre_trainneural(parser_std)
model.save_weights('./my_model.h5')

retorno = parse(parser_std, model, "./corpus/test/portuguese_test.conll")

arq = open("./retorno_parser.txt", "w")
for graph in retorno:
    arq.write(graph.to_conll(style=10))
arq.close()
