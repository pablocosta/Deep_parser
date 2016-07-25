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
from nltk.parse import DependencyGraph
from parser import TransitionParser
from sklearn.metrics import recall_score, precision_score
from load_model import Model

parser_std = TransitionParser('arc-standard')
model_ = Model(None, "./corpus/test/portuguese_test.conll", "./my_model_pt.h5", parser_std)
model_.load_weights()
model_.parse_language()

model_.evaluate_parser()
