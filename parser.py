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
import pickle
from DepedencyTree import DependencyTree

try:
    from numpy import array,argmax, delete
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

    def __init__(self, sent):
        """
        :param dep_graph: the representation of an input in the form of dependency graph.
        :type dep_graph: DependencyGraph where the dependencies are not specified.
        """
        # dep_graph.nodes contain list of token for a sentence
        self.stack = [0]  # The root element
        self.buffer = []
        self.tree = DependencyTree()
        i=1
        while i <= sent.n:
            self.buffer.append(i)
            self.tree.add(-1, "UNKNOWN")
            i+=1
        self.arcs = []  # empty set of arc
        self._tokens = sent.words
        self.n = sent.n
        self.sent = sent
        self._max_address = sent.n

    def get_head(self, k):
        return self.tree.get_head(k)
    def getLabel(self, k):
        return self.tree.get_label(k)
    def remove_second_top_stack(self):
        n_stack = self.get_stack_size()
        if n_stack < 2:
            return False
        del self.stack[self.get_stack_size()-2]
        return True
    def shift(self):
        """
        Note that the algorithm for shift is the SAME for arc-standard and arc-eager
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """

        k = self.getBuffer(0)
        if k == -1:
            return False
        del self.buffer[0]
        self.stack.append(k)
        return True

    def remove_top_stack(self):
        n_stack = self.get_stack_size()
        if n_stack < 1:
            return False
        del self.stack[ self.get_stack_size() -1]
        return True
    def __str__(self):
        return 'Stack : ' + \
               str(self.stack) + '  Buffer : ' + str(self.buffer) + '   Arcs : ' + str(self.arcs)
    def has_other_child(self,k, depgraph):
        i=1
        while i <= self.tree.n:
            if depgraph.get_head(i) == k and self.tree.get_head(i) != k:
                return True
            i +=1
        return False
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
        if (k < 0) or (k > self.tree.n):
            return -1
        i = 1
        c = 0
        while (i < k):
            if self.tree.get_head(i) == k:
                c += 1
                if c == cnt:
                    return i
            i += 1
        return -1

    def getLeftChild(self, k):
        return self.getLeftChild_(k, 1)

    def getRightChild_(self, k, cnt):
        if (k < 0) or (k > self.tree.n):
            return -1
        c = 0
        i = self.tree.n
        while i > k:
            if self.tree.get_head(i) == k:

                c += 1
                if c == cnt:
                    return i
            i -= 1

        return -1
    def getRightChild(self, k):
        return self.getRightChild_(k, 1)
    def get_buffer_size(self):
        return len(self.buffer)
    def get_stack_size(self):
        return len(self.stack)

    def getWord(self, k):
        if k == 0:
            return "ROOT"
        else:
            k -= 1

        if (k < 0) or (k >= self.sent.n):
            return "NULL"
        else:
            return self.sent.words[k]

    def getPos(self, k):
        if k == 0:
            return "ROOT"
        else:
            k -= 1

        if (k < 0) or (k >= self.sent.n):
            return "NULL"
        else:
            return self.sent.poss[k]
    def add_arc(self, h, m, l):
        self.tree.set(m,h,l)

    def getStack(self, k):
        n_stack = len(self.stack)
        if k >= 0 and k < n_stack:
            return self.stack[n_stack - 1 - k]
        else:
            return -1

    def getBuffer(self, k):
        n_buffer = len(self.buffer)
        if (k >= 0) and (k < n_buffer):
            return self.buffer[k]
        else:
            return -1

    def extract_features(self):
        """
        Extract the set of features for the current configuration. Implement standard features from original describe by Joakin Nivre.
        :return: 3 lists(str) from the features
        """

        # Get word and PoS from stak
        word_features = []
        pos_features = []
        label_features = []
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

            # leftmost child
            index = self.getLeftChild(k)
            word_features.append(self.getWord(index))
            pos_features.append(self.getPos(index))
            label_features.append(self.getLabel(index))

            # rightmost child
            index = self.getRightChild(k)
            word_features.append(self.getWord(index))
            pos_features.append(self.getPos(index))
            label_features.append(self.getLabel(index))

            # second leftmost child
            index = self.getLeftChild_(k, 2)
            word_features.append(self.getWord(index))
            pos_features.append(self.getPos(index))
            label_features.append(self.getLabel(index))

            # second rightmost child
            index = self.getRightChild_(k, 2)
            word_features.append(self.getWord(index))
            pos_features.append(self.getPos(index))
            label_features.append(self.getLabel(index))

            # left-leftmostchild
            index = self.getLeftChild(self.getLeftChild(k))
            word_features.append(self.getWord(index))
            pos_features.append(self.getPos(index))
            label_features.append(self.getLabel(index))

            # right rightmostchild
            index = self.getRightChild(self.getRightChild(k))
            word_features.append(self.getWord(index))
            pos_features.append(self.getPos(index))
            label_features.append(self.getLabel(index))

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

        k = conf.get_buffer(0)
        if k == -1:
            return False
        conf.buffer.pop(0)
        conf.stack.append(k)
        return True



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
        if not (algorithm in [self.ARC_STANDARD, self.ARC_EAGER]):
            raise ValueError(" Currently we only support %s and %s " %
                             (self.ARC_STANDARD, self.ARC_EAGER))
        self._algorithm = algorithm

        self._dictionary = {}
        self._transition = {}
        self._match_transition = {}

    def _get_dep_relation(self, depgraph, conf):
        w1 = conf.getStack(1)
        w2 = conf.getStack(0)
        if w1 > 0 and depgraph.get_head(w1) == w2:
            return Transition.LEFT_ARC+":"+str(depgraph.get_label(w1))
        elif (w1 >= 0) and (depgraph.get_head(w2) ==w1) and (not(conf.has_other_child(w2, depgraph))):
            return Transition.RIGHT_ARC+":"+str(depgraph.get_label(w2))
        else:
            return Transition.SHIFT

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

    def _write_to_file(self, key, w_features, t_features, l_features):
        """
        write the binary features to input file and update the transition dictionary
        """

        self._transition.setdefault(key, len(self._transition) + 1)
        self._match_transition[self._transition[key]] = key

        return key, w_features, t_features, l_features

    def _write_blenk_in_file(self, input_file):
        input_file.write("\n")
    def is_terminal(self, conf):

        if conf.get_stack_size() == 1 and conf.get_buffer_size() == 0:
            return True
        else:
            return False
    def get_udt_sub_obj_relations(self):
        udt_sub_obj_relations = ["dobj","iobj","adpobj","csubj",
                                      "csubjpass", "nsubj", "nsubjpass"]
        return udt_sub_obj_relations

    def get_ponctuation_tags(self):
        en_punct_tags = ["``","''", ".",",",":"]
        return en_punct_tags

    def evaluate(self, sents, pred_trees, gold_trees):
        punc_tags = self.get_ponctuation_tags()
        sub_obj_relations = self.get_udt_sub_obj_relations()
        correct_arcs = 0
        correct_arcs_wo_punc = 0
        correct_heads = 0
        correct_heads_wo_punc = 0

        correct_heads_sub_obj = 0
        sum_arcs_sub_obj = 0

        correct_trees = 0
        correct_trees_wo_punc = 0
        correct_root = 0

        sum_arcs = 0
        sum_arcs_wo_punc = 0


        for i in range(0, len(pred_trees)):
            n_correct_head = 0
            n_correct_head_wo_punc = 0
            non_punc = 0
            for j in range(1, pred_trees[i].n):
                if pred_trees[i].get_head(j) == gold_trees[i].get_head(j):
                    correct_heads  += 1
                    n_correct_head += 1
                    if pred_trees[i].get_label(j) == gold_trees[i].get_label(j):
                        correct_arcs += 1
                sum_arcs += 1
                tag = sents[i].poss[j-1]

                if tag not in punc_tags:
                    sum_arcs_wo_punc += 1
                    non_punc += 1
                    if pred_trees[i].get_head(j) == gold_trees[i].get_head(j):
                        correct_heads_wo_punc += 1
                        n_correct_head_wo_punc += 1
                        if pred_trees[i].get_label(j) == gold_trees[i].get_label(j):
                            correct_arcs_wo_punc += 1

                gold_rel = gold_trees[i].get_label(j)

                if gold_rel in sub_obj_relations:
                    sum_arcs_sub_obj += 1
                    if pred_trees[i].get_head(j) == gold_trees[i].get_head(j):
                        correct_heads_sub_obj += 1

            if n_correct_head == pred_trees[i].n:
                correct_trees += 1
            if non_punc == n_correct_head_wo_punc:
                correct_trees_wo_punc +=1
            if pred_trees[i].get_root() == gold_trees[i].get_root():
                correct_root += 1
        print("UAS: ")
        print(correct_heads * 100.0 / sum_arcs)
        print("-------")
        print("UASwoPunc: ")
        print(correct_heads_wo_punc * 100.0 / sum_arcs_wo_punc)
        print("-------")
        print("LAS: ")
        print(correct_arcs  * 100.0 / sum_arcs)
        print("-------")
        print("LASwoPunc: ")
        print(correct_arcs_wo_punc  * 100.0 / sum_arcs_wo_punc)
        print("-------")


    def apply(self, conf, op):
        w1 = conf.getStack(1)
        w2 = conf.getStack(0)
        label = ":"

        if op[0].startswith("L"):
            label = ":".join(op[1:])
            conf.add_arc(w2,w1, label)
            conf.remove_second_top_stack()

        elif op[0].startswith("R"):
            label = ":".join(op[1:])
            conf.add_arc(w1,w2,label)
            conf.remove_top_stack()
        else:
            conf.shift()


    def can_apply(self, conf, op):
        if op[0].startswith("L") or op[0].startswith("R"):
            label = op[1]

            if op[0].startswith("L"):
                h = conf.getStack(0)
            else:
                h = conf.getStack(1)
            if h <0:
                return False
            if h == 0 and not("ROOT"in label.upper()):
                return False
            #ver se deixa comentado ou nao multirooted
            if h > 0 and ("ROOT" in label.upper()):
                return False
        n_stack  = conf.get_stack_size()
        n_buffer = conf.get_buffer_size()
        if op[0].startswith("L"):
            return n_stack > 2
        elif op[0].startswith("R"):
            return n_stack > 2 or (n_stack == 2 and n_buffer == 0)
        else:
            return n_buffer > 0

    def _create_training_examples_arc_std(self, sents, tree, input_file):
        """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : Page 32, Chapter 3. Dependency Parsing by Sandra Kubler, Ryan McDonal and Joakim Nivre (2009)
        """
        operation = Transition(self.ARC_STANDARD)
        count_proj = 0
        training_seq = []

        for i in range(0, len(sents)):
            if tree[i].is_projective():
                count_proj += 1
                conf = Configuration(sents[i])
                while not self.is_terminal(conf):
                    (w_features, p_features, l_features) = conf.extract_features()
                    # binary_features = self._convert_to_binary_features(features)

                    for n in range(0, len(l_features)):
                        if l_features[n] == -1:
                            l_features[n] = "NULL"
                    # Left-arc operation

                    rel = self._get_dep_relation(tree[i], conf)


                    baseTransition = rel.split(":")
                    training_seq.append(self._write_to_file(rel, w_features, p_features, l_features))
                    self.apply(conf, baseTransition)




        pickle.dump(training_seq, input_file)
        print(" Number of training examples : " + str(len(sents)))
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
    def Get_features_(self, list, dict):

        return [dict[l] for l in list]

    def _can_apply(self, conf, dict_op, sort_dict, max):

        for k in range(0,max):
            for key in dict_op.keys():
                if dict_op[key] == sort_dict[k]:
                    strTransition = key
                    op = strTransition.split(":")
                    if self.can_apply(conf, op):
                        return strTransition
        return " "




    def predict(self,sent, model, words, tags, labels, dict_op):
        conf = Configuration(sent)

        while not self.is_terminal(conf):

            (word_features, pos_features, label_features) = conf.extract_features()
            for n in range(0, len(label_features)):
                if label_features[n] == -1:
                    label_features[n] = "NULL"
            y_pred_model = model.predict_proba(X=[array([self.Get_features_(word_features,words)
                                                          ]), array([self.Get_features_(pos_features,tags)]
                                                                    ), array([self.Get_features_(label_features, labels)]
                                                                             )])
            sort_dict = dict()
            max = len(y_pred_model[0])
            y = y_pred_model[0]
            for ind in range(0,max):
                sort_dict[ind] = argmax(y)
                y = delete(y,sort_dict[ind])
            strTransition = self._can_apply(conf, dict_op, sort_dict, max)

            baseTransition = strTransition.split(":")

            self.apply(conf, baseTransition)
        return conf.tree
        # Finish with operations build the dependency graph from Conf.arcs


    def parse(self, sents, tree, model, words, tags, labels, dict_op):
        """
        :param depgraphs: the list of test sentence, each sentence is represented as a dependency graph where the 'head' information is dummy
        :type depgraphs: list(DependencyGraph)
        :param modelfile: the model file
        :type modelfile: str
        :return: list (DependencyGraph) with the 'head' and 'rel' information
        """
        operation = Transition(self._algorithm)

        predicted_tree = []
        for i in range(0, len(sents)):
            print("Arvore:",i)
            predicted_tree.append(self.predict(sents[i],  model, words, tags, labels, dict_op))

        self.evaluate(sents, predicted_tree, tree)

        return predicted_tree
