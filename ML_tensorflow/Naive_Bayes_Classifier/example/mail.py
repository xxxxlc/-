"""
mail classification problem
"""

import numpy as np
import pandas as pd

from ML_tensorflow.Naive_Bayes_Classifier.NaiveBayesClassifier import NaiveBayesClassifier
from ML_tensorflow.Naive_Bayes_Classifier.AODE import AODE
from ML_tensorflow.dataloader.loader import Loader
from ML_tensorflow.Naive_Bayes_Classifier.NBC import NBC
from ML_tensorflow.Naive_Bayes_Classifier.TAN import TAN

data_path = 'D:\\workplace\\ML\\ML_tensorflow\\Naive_Bayes_Classifier\\datasample\\mail.xlsx'
loader = Loader()
X, y = loader.file_reader.read_excel(data_path)

a = NaiveBayesClassifier(X, y)
a.train()
a.classify_one([1, 1, 1, 1], True)
a.evaluation(a.train_label, a.train_y_pred)

a.train_skl('MNB')
a.evaluation(a.train_label, a.train_y_pred)

b = AODE(X, y)
y_pred = b.classify(X)
b.evaluation(y, y_pred[:, 1])

c = NBC(X, y, [], [], [0, 1, 2, 3], [])
y_pred = c.classify(X)
c.evaluation(y, y_pred[:, 1])

d = TAN(X, y, [], [], [0, 1, 2, 3], [])
y_pred = d.classify(X)
d.evaluation(y, y_pred[:, 1])
