import keras
import numpy as np
import os
import copy
import pickle
from os import path
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.core import Dense, Activation, Dropout, Merge, Reshape
from keras.layers.embeddings import Embedding
from parser import TransitionParser
from sklearn.metrics import recall_score, precision_score
from load_model import Model
from DepedencyTree import ReadTrees



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
    batch_size = 10000
    nb_epoch = 100

    """ TO-DO
        fazer funcao X^3
    """

    words = Load_embedding_file("word.txt")
    tags = Load_embedding_file("pos.txt")
    labels = Load_embedding_file("label.txt")
    #make_softmax_arq(labels)

    dirs = [os.listdir("./corpus/test"), os.listdir("./corpus/train")]


    make_softmax_arq(labels)
    output_dim = (len(labels.keys())-1)*2 + 1

    # create dictionary for the index
    words_indx = create_dict_inx(words)
    tags_indx = create_dict_inx(tags)
    labels_indx = create_dict_inx(labels)

    w_weights = return_weights(words, words_indx)
    p_weights = return_weights(tags, tags_indx)
    l_weights = return_weights(labels, labels_indx)


    word_features = Sequential()
    word_features.add(Embedding(input_dim=len(words.keys())+1,input_length=18, output_dim=50,  mask_zero=True, weights=[w_weights]))

    pos_features = Sequential()
    pos_features.add(Embedding(input_dim=len(tags.keys())+1,input_length=18, output_dim=50, mask_zero=True, weights=[p_weights]))

    label_features = Sequential()
    label_features.add(Embedding(input_dim=len(labels.keys())+1,input_length=12, output_dim=50,  mask_zero=True, weights=[l_weights]))

    model = Sequential()

    model.add(Merge([word_features, pos_features, label_features], mode='concat', concat_axis = 1))
    model.add(Reshape((48*50,)))
    model.add(Dropout((0.5)))
    model.add(Dense(output_dim=200, activation="relu",W_regularizer=l2(1e-6)))
    model.add(Dense(output_dim=200, input_dim=200, activation="relu",W_regularizer=l2(1e-6)))
    # To-do: Modelar a ativacao tripla X 3
    
    model.add(Dense(output_dim=output_dim, input_dim=200))
    model.add(Activation('softmax'))
    adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad)
    # Extract one-hot repersentations

    (dict_op, one_hot) = to_one_hot(open("./corpus/temp/softmax_arq.txt"))
    save_dict(dict_op)
    dict_opaux = copy.deepcopy(dict_op)
    for element in range(0, len(one_hot[0])):
        for key in dict_op.keys():
            if dict_opaux[key] == element:
                dict_op[key] = one_hot[0][element]


    extract_training_data_arc_standard(parser_std)
    X, Y = get_all(words_indx, tags_indx, labels_indx, dict_op)
    Y = np.array(Y)
    model.fit(X=[np.array(X[0]), np.array(X[1]), np.array(X[2])], y=Y, nb_epoch=nb_epoch,batch_size=batch_size,validation_split=0.1)

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

def save_dict(dict):
    input_file = open("./corpus/temp/dict_onehot.pickl", "wb")
    pickle.dump(dict, input_file)
    input_file.close()

def load_dict():
    a = open("./corpus/temp/dict_onehot.pickl", 'rb')
    object = pickle.load(a)

    return object
def Get_features_fromop(list_fatures, dict_op):

    return np.array(dict_op[list_fatures])

def Get_features_(list, dict):

    return [dict[l] for l in list]



def get_all(words, tags, labels, dict_op):
    # this function get all configurations from the training files
    X_word = []
    X_tags = []
    X_label = []
    Y_ = []

    # Todo o conjunto de treinamento esta no arquivo input_file da forma Y [features_word] [features_pos] [features_label]
    #input_file = open(, "rb")

    a = open("./corpus/temp/parc_test.pickl", 'rb')
    object = pickle.load(a)
    for element in object:
        Y_.append(Get_features_fromop(element[0], dict_op))
        X_word.append(np.array(Get_features_(element[1], words)))
        X_tags.append(np.array(Get_features_(element[2], tags)))
        X_label.append(np.array(Get_features_(element[3], labels)))



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
    a.close()
    return [X_word, X_tags, X_label], Y_


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
            if op == 0 and not("NULL"in l):
                operations_arq.write("LEFTARC" + ":" + l + "\n")

            elif op == 1 and not("NULL"in l):
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


def extract_training_data_arc_standard(parser_std):
    trees = ReadTrees("./corpus/train/english_train.conll")
    tr,se = trees.load_corpus()
    input_file = open("./corpus/temp/parc_test.pickl", "wb")
    parser_std._create_training_examples_arc_std(se,tr, input_file)
    input_file.close()
    #input_file.close()



def f_measure(Y_pred, Y):
    r = recall_score(Y_pred, Y, average="macro")
    p = precision_score(Y_pred, Y, average="macro")
    f = 2 * (p * r) / (p + r)
    return f
# carrega analisador
parser_std = TransitionParser('arc-standard')

# Treinamento do modelo
model = pre_trainneural(parser_std)
model.save_weights('./my_model2.h5', overwrite=True)

model_ = Model(model, "./corpus/test/english_test.conll", "./my_model2.h5", parser_std)
model_.parse_language()
