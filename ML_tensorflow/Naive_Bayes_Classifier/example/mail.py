"""
mail classification problem
"""

import numpy as np
import pandas as pd

from ML_tensorflow.Naive_Bayes_Classifier.NaiveBayesClassifier import NaiveBayesClassifier

data_path = 'D:\\workplace\\ML\\ML_tensorflow\\Naive_Bayes_Classifier\\datasample\\mail.xlsx'
df = pd.read_excel(data_path)
category = df.columns
X = df.drop([category[0], category[-1]], axis=1)
X = np.array(X)
y = np.array(df[category[-1]])

a = NaiveBayesClassifier(X, y)
a.train()
a.classify_one([1, 1, 1, 1], True)
a.laplace_smoothing()
a.classify_one([1, 1, 1, 1], True)
a.evaluation(a.train_label, a.train_y_pred)