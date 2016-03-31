# Natural Language Toolkit: Arc-Standard and Arc-eager Transition Based Parsers
#
# Author: Long Duong <longdt219@gmail.com>
#
# Copyright (C) 2001-2015 NLTK Project
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy.ma as ma
import copy
from copy import deepcopy
try:
    from numpy import array
    from scipy import sparse
except ImportError:
    pass

from ParserI import ParserI



class Configuration(object):
    """
    Class for holding configuration which is the partial analysis of the input sentence.
    The transition based parser aims at finding set of operators that transfer the initial
    configuration to the terminal configuration.

    The configuration includes:
        - Stack: for storing partially proceeded words
        - Buffer: for storing remaining input words
        - Set of arcs: for storing partially built dependency tree

    This class also provides a method to represent a configuration as list of features.
    """

    def __init__(self, dep_graph):
        """
        :param dep_graph: the representation of an input in the form of dependency graph.
        :type dep_graph: DependencyGraph where the dependencies are not specified.
        """
        # dep_graph.nodes contain list of token for a sentence
        self.stack = [0]  # The root element
        self.buffer = list(range(1, len(dep_graph.nodes)))  # The rest is in the buffer
        self.arcs = []  # empty set of arc
        self._tokens = copy.deepcopy(dep_graph.nodes)
        self.n= len(dep_graph.nodes)
        self._max_address = len(self.buffer)

    def __str__(self):
        return 'Stack : ' + \
            str(self.stack) + '  Buffer : ' + str(self.buffer) + '   Arcs : ' + str(self.arcs)

    def _check_informative(self, feat, flag=False):
        """
        Check whether a feature is informative
        The flag control whether "_" is informative or not
        """
        if feat is None:
            return False
        if feat == '':
            return False
        if flag is False:
            if feat == '_':
                return False
        return True

    def getLeftChild_(self, k, cnt):
        if (k < 0) or (k > self.n):
            return -1
        i=1
        c =0
        while (i < k):
            aux = self._tokens[i]
            if aux['head'] == k:
                c += 1
                if c == cnt:
                    return i
            i += 1
        return -1

    def getLeftChild (self, i):
        return self.getLeftChild_(i, 1)


    def getRightChild_(self, k, cnt):
        if (k < 0) or (k > self.n):
            return -1
        c = 0
        i = self.n
        while i > k:
            aux = self._tokens[i]
            if aux['head'] == k:

                c += 1
                if c == cnt:
                    return i
            i -= 1


        return -1


    def getRightChild(self, i):
        return self.getRightChild_(i, 1)

    def getWord(self,k):
        if k == 0:
            return "-ROOT-"
        else:
            k -= 1

        if (k < 0) or (k >= len(self.buffer)):
            return "-NULL-"
        else:
            token = self._tokens[k]
            return token['word']

    def getPos(self,k):
        if k == 0:
            return "-ROOT-"
        else:
            k -= 1

        if (k < 0) or (k >= len(self.buffer)):
            return "-NULL-"
        else:
            token = self._tokens[k]
            return token['tag']

    def getLabel(self, k):

        if k <= 0 or k > self.n:
            return "-NULL-"
        token = self._tokens[k]
        return token['rel']



    def getStack(self, indice):
        n_stack = len(self.stack)
        if indice >=0 and indice < n_stack:
            return self.stack[n_stack - 1 - indice]
        else:
            return -1

    def getBuffer(self, indice):
        n_buffer = len(self.buffer)
        if ((indice >= 0) and (indice < n_buffer)):
            return self.buffer[indice]
        else:
            return -1



    def extract_features(self):
        """
        Extract the set of features for the current configuration. Implement standard features from original describe by Joakin Nivre.
        :return: 3 lists(str) from the features
        """

        #Get word and PoS from stak
        word_features   = []
        pos_features    = []
        label_features  = []
        for i in reversed(range(0, 3)):
            stack_idx0 = self.getStack(i)
            word_features.append(self.getWord(stack_idx0))
            pos_features.append(self.getPos(stack_idx0))


        for i in range(0, 3):
            buffer_idx0 = self.getBuffer(i)
            word_features.append(self.getWord(buffer_idx0))
            pos_features.append(self.getPos(buffer_idx0))

        for i in range(0, 2):
            k = self.getStack(i)

            #leftmost child
            index = self.getLeftChild(k)
            word_features.append(self.getWord(index))
            pos_features.append(self.getPos(index))
            label_features.append(self.getLabel(index))


            #rightmost child
            index = self.getRightChild(k)
            word_features.append(self.getWord(index))
            pos_features.append(self.getPos(index))
            label_features.append(self.getLabel(index))

            #second leftmost child
            index = self.getLeftChild_(k, 2)
            word_features.append(self.getWord(index))
            pos_features.append(self.getPos(index))
            label_features.append(self.getLabel(index))

            #second rightmost child
            index = self.getRightChild_(k, 2)
            word_features.append(self.getWord(index))
            pos_features.append(self.getPos(index))
            label_features.append(self.getLabel(index))

            # left-leftmostchild
            index = self.getLeftChild(self.getLeftChild(k))
            word_features.append(self.getWord(index))
            pos_features.append(self.getPos(index))
            label_features.append(self.getLabel(index))

            #right rightmostchild
            index = self.getRightChild(self.getRightChild(k))
            word_features.append(self.getWord(index))
            pos_features.append(self.getPos(index))
            label_features.append(self.getLabel(index))

        print("tamanho word:", len(word_features))
        print("tamanho pos:", len(pos_features))
        print("tamanho label:", len(label_features))
        return word_features, pos_features, label_features


class Transition(object):
    """
    This class defines a set of transition which is applied to a configuration to get another configuration
    Note that for different parsing algorithm, the transition is different.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self, alg_option):
        """
        :param alg_option: the algorithm option of this parser. Currently support `arc-standard` and `arc-eager` algorithm
        :type alg_option: str
        """
        self._algo = alg_option
        if alg_option not in [
                TransitionParser.ARC_STANDARD,
                TransitionParser.ARC_EAGER]:
            raise ValueError(" Currently we only support %s and %s " %
                                        (TransitionParser.ARC_STANDARD, TransitionParser.ARC_EAGER))

    def left_arc(self, conf, relation):
        """
        Note that the algorithm for left-arc is quite similar except for precondition for both arc-standard and arc-eager
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if (len(conf.buffer) <= 0) or (len(conf.stack) <= 0):
            return -1
        if conf.buffer[0] == 0:
            # here is the Root element
            return -1

        idx_wi = conf.stack[len(conf.stack) - 1]

        flag = True
        if self._algo == TransitionParser.ARC_EAGER:
            for (idx_parent, r, idx_child) in conf.arcs:
                if idx_child == idx_wi:
                    flag = False

        if flag:
            conf.stack.pop()
            idx_wj = conf.buffer[0]
            conf.arcs.append((idx_wj, relation, idx_wi))
        else:
            return -1

    def right_arc(self, conf, relation):
        """
        Note that the algorithm for right-arc is DIFFERENT for arc-standard and arc-eager
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if (len(conf.buffer) <= 0) or (len(conf.stack) <= 0):
            return -1
        if self._algo == TransitionParser.ARC_STANDARD:
            idx_wi = conf.stack.pop()
            idx_wj = conf.buffer[0]
            conf.buffer[0] = idx_wi
            conf.arcs.append((idx_wi, relation, idx_wj))
        else:  # arc-eager
            idx_wi = conf.stack[len(conf.stack) - 1]
            idx_wj = conf.buffer.pop(0)
            conf.stack.append(idx_wj)
            conf.arcs.append((idx_wi, relation, idx_wj))

    def reduce(self, conf):
        """
        Note that the algorithm for reduce is only available for arc-eager
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """

        if self._algo != TransitionParser.ARC_EAGER:
            return -1
        if len(conf.stack) <= 0:
            return -1

        idx_wi = conf.stack[len(conf.stack) - 1]
        flag = False
        for (idx_parent, r, idx_child) in conf.arcs:
            if idx_child == idx_wi:
                flag = True
        if flag:
            conf.stack.pop()  # reduce it
        else:
            return -1

    def shift(self, conf):
        """
        Note that the algorithm for shift is the SAME for arc-standard and arc-eager
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if len(conf.buffer) <= 0:
            return -1
        idx_wi = conf.buffer.pop(0)
        conf.stack.append(idx_wi)


class TransitionParser(ParserI):

    """
    Class for transition based parser. Implement 2 algorithms which are "arc-standard" and "arc-eager"
    """

    ARC_STANDARD = 'arc-standard'
    ARC_EAGER = 'arc-eager'

    def __init__(self, algorithm):
        """
        :param algorithm: the algorithm option of this parser. Currently support `arc-standard` and `arc-eager` algorithm
        :type algorithm: str
        """
        if not(algorithm in [self.ARC_STANDARD, self.ARC_EAGER]):
            raise ValueError(" Currently we only support %s and %s " %
                                        (self.ARC_STANDARD, self.ARC_EAGER))
        self._algorithm = algorithm

        self._dictionary = {}
        self._transition = {}
        self._match_transition = {}

    def _get_dep_relation(self, idx_parent, idx_child, depgraph):
        p_node = depgraph.nodes[idx_parent]
        c_node = depgraph.nodes[idx_child]

        if c_node['word'] is None:
            return None  # Root word

        if c_node['head'] == p_node['address']:
            return c_node['rel']
        else:
            return None

    def _convert_to_binary_features(self, features):
        """
        :param features: list of feature string which is needed to convert to binary features
        :type features: list(str)
        :return : string of binary features in libsvm format  which is 'featureID:value' pairs
        """
        unsorted_result = []
        for feature in features:
            self._dictionary.setdefault(feature, len(self._dictionary))
            unsorted_result.append(self._dictionary[feature])

        # Default value of each feature is 1.0
        return ' '.join(str(featureID) + ':1.0' for featureID in sorted(unsorted_result))

    def _is_projective(self, depgraph):
        arc_list = []
        for key in depgraph.nodes:
            node = depgraph.nodes[key]

            if 'head' in node:
                childIdx = node['address']
                parentIdx = node['head']
                if parentIdx is not None:
                    arc_list.append((parentIdx, childIdx))

        for (parentIdx, childIdx) in arc_list:
            # Ensure that childIdx < parentIdx
            if childIdx > parentIdx:
                temp = childIdx
                childIdx = parentIdx
                parentIdx = temp
            for k in range(childIdx + 1, parentIdx):
                for m in range(len(depgraph.nodes)):
                    if (m < childIdx) or (m > parentIdx):
                        if (k, m) in arc_list:
                            return False
                        if (m, k) in arc_list:
                            return False
        return True

    def _write_to_file(self, key, w_features, t_features, l_features, input_file):
        """
        write the binary features to input file and update the transition dictionary
        """
        self._transition.setdefault(key, len(self._transition) + 1)
        self._match_transition[self._transition[key]] = key
        input_str = str(key) + ' ' + str(w_features)+" "+ str(t_features)+" "+str(l_features) + '\n'

        input_file.write(input_str)
    def _write_blenk_in_file(self, input_file):
        input_file.write("\n")

    def _create_training_examples_arc_std(self, depgraphs, input_file):
        """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : Page 32, Chapter 3. Dependency Parsing by Sandra Kubler, Ryan McDonal and Joakim Nivre (2009)
        """
        operation = Transition(self.ARC_STANDARD)
        count_proj = 0
        training_seq = []
        i = 0
        training_seq.append(dict())
        for depgraph in depgraphs:
            if not self._is_projective(depgraph):
                continue

            count_proj += 1
            conf = Configuration(depgraph)

            while len(conf.buffer) > 0:
                b0 = conf.buffer[0]
                (w_features, p_features, l_features) = conf.extract_features()
                #binary_features = self._convert_to_binary_features(features)



                if len(conf.stack) > 0:
                    s0 = conf.stack[len(conf.stack) - 1]

                    # Left-arc operation
                    rel = self._get_dep_relation(b0, s0, depgraph)

                    if rel is not None:
                        key = Transition.LEFT_ARC + ':' + rel

                        self._write_to_file(key, w_features, p_features, l_features, input_file)
                        operation.left_arc(conf, rel)
                        continue

                    # Right-arc operation
                    rel = self._get_dep_relation(s0, b0, depgraph)
                    if rel is not None:
                        precondition = True
                        # Get the max-index of buffer
                        maxID = conf._max_address

                        for w in range(maxID + 1):
                            if w != b0:
                                relw = self._get_dep_relation(b0, w, depgraph)

                                if relw is not None:
                                    if (b0, relw, w) not in conf.arcs:
                                        precondition = False

                        if precondition:
                            key = Transition.RIGHT_ARC + ':' + rel
                            self._write_to_file(key, w_features, p_features, l_features, input_file)
                            operation.right_arc(conf, rel)
                            continue

                # Shift operation as the default
                key = Transition.SHIFT
                self._write_to_file(key,w_features, p_features, l_features, input_file)
                operation.shift(conf)

        input_file.close()
        print(" Number of training examples : " + str(len(depgraphs)))
        print(" Number of valid (projective) examples : " + str(count_proj))

    def _create_training_examples_arc_eager(self, depgraphs, input_file):
        """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : 'A Dynamic Oracle for Arc-Eager Dependency Parsing' by Joav Goldberg and Joakim Nivre
        """
        operation = Transition(self.ARC_EAGER)
        countProj = 0
        training_seq = []

        for depgraph in depgraphs:
            if not self._is_projective(depgraph):
                continue

            countProj += 1
            conf = Configuration(depgraph)
            while len(conf.buffer) > 0:
                b0 = conf.buffer[0]
                features = conf.extract_features()
                binary_features = self._convert_to_binary_features(features)



                if len(conf.stack) > 0:
                    s0 = conf.stack[len(conf.stack) - 1]
                    # Left-arc operation
                    rel = self._get_dep_relation(b0, s0, depgraph)
                    if rel is not None:
                        key = Transition.LEFT_ARC + ':' + rel
                        self._write_to_file(key, binary_features, input_file)
                        operation.left_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # Right-arc operation
                    rel = self._get_dep_relation(s0, b0, depgraph)
                    if rel is not None:
                        key = Transition.RIGHT_ARC + ':' + rel
                        self._write_to_file(key, binary_features, input_file)
                        operation.right_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # reduce operation
                    flag = False
                    for k in range(s0):
                        if self._get_dep_relation(k, b0, depgraph) is not None:
                            flag = True
                        if self._get_dep_relation(b0, k, depgraph) is not None:
                            flag = True
                    if flag:
                        key = Transition.REDUCE
                        self._write_to_file(key, binary_features, input_file)
                        operation.reduce(conf)
                        training_seq.append(key)
                        continue

                # Shift operation as the default
                key = Transition.SHIFT
                self._write_to_file(key, binary_features, input_file)
                operation.shift(conf)
                training_seq.append(key)

        print(" Number of training examples : " + str(len(depgraphs)))
        print(" Number of valid (projective) examples : " + str(countProj))
        return training_seq

    def parse(self, depgraphs, model, words, tags, labels, dict_op):
        """
        :param depgraphs: the list of test sentence, each sentence is represented as a dependency graph where the 'head' information is dummy
        :type depgraphs: list(DependencyGraph)
        :param modelfile: the model file
        :type modelfile: str
        :return: list (DependencyGraph) with the 'head' and 'rel' information
        """
        result = []
        operation = Transition(self._algorithm)
        for depgraph in depgraphs:

            dictnionary_graph = depgraph.nodes
            conf = Configuration(depgraph)
            while len(conf.buffer) > 0:
                b0 = conf.buffer[0]

                if len(conf.stack) > 0:


                    s0 = conf.stack[len(conf.stack)-1]
                    w1_new = (dictnionary_graph[s0])["word"]
                    pos = (dictnionary_graph[s0])["ctag"]
                    label = (dictnionary_graph[s0])["rel"]
                    w2_new = (dictnionary_graph[b0])["word"]
                    # It's best to use decision function as follow BUT it's not supported yet for sparse SVM
                    # Using decision funcion to build the votes array
                    #dec_func = model.decision_function(x_test)[0]
                    #votes = {}
                    #k = 0
                    # for i in range(len(model.classes_)):
                    #    for j in range(i+1, len(model.classes_)):
                    #        #if  dec_func[k] > 0:
                    #            votes.setdefault(i,0)
                    #            votes[i] +=1
                    #        else:
                    #           votes.setdefault(j,0)
                    #           votes[j] +=1
                    #        k +=1
                    # Sort votes according to the values
                    #sorted_votes = sorted(votes.items(), key=itemgetter(1), reverse=True)

                    #extract the right X configuration
                    w1 = words[str(w1_new)]
                    w2 = words[str(w2_new)]
                    pos1 = tags[pos]
                    label1 = labels[str(label)]

                    concat = array([ma.concatenate([w1, w2, pos1, label1])])

                    y_pred_model = model.predict_classes(concat)

                    for key in dict_op.keys():
                        if dict_op[key] == y_pred_model[0]:
                            strTransition = key

                    # Note that SHIFT is always a valid operation

                    #pegar o y correto key?
                    baseTransition = strTransition.split(":")[0]

                    if baseTransition == Transition.LEFT_ARC:
                        if operation.left_arc(conf, strTransition.split(":")[1]) != -1:
                            break
                    elif baseTransition == Transition.RIGHT_ARC:
                        if operation.right_arc(conf, strTransition.split(":")[1]) != -1:
                            break
                    elif baseTransition == Transition.REDUCE:
                        if operation.reduce(conf) != -1:
                            break
                    elif baseTransition == Transition.SHIFT:
                        if operation.shift(conf) != -1:
                            break




                operation.shift(conf)

            # Finish with operations build the dependency graph from Conf.arcs

            new_depgraph = deepcopy(depgraph)
            for key in new_depgraph.nodes:
                node = new_depgraph.nodes[key]
                node['rel'] = ''
                # With the default, all the token depend on the Root
                node['head'] = 0
            for (head, rel, child) in conf.arcs:
                c_node = new_depgraph.nodes[child]
                c_node['head'] = head
                c_node['rel'] = rel
            result.append(new_depgraph)

        return result
