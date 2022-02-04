"""
Naive Bayes Classifier
"""

import numpy as np
import pandas as pd
import copy

from ML_tensorflow.dataloader.loader import Loader
from ML_tensorflow.evaluator.evaluator import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from functools import reduce
from collections import Counter


class NaiveBayesClassifier(object):
    """
    Base Naive Bayes Classifier
    function:
        Currently supported discrete feature variables
    """
    def __init__(self, train_data, train_label, test_data=None, test_label=None) -> None:
        """
        build train and test dataset,
        build priori probability,
        build conditional probability.
        ----------------------------------
        input label data must be list of integer:
        for example:
            two labels: [0, 1]
            three labels: [0, 1, 2]
        as same as feature:
            two features: [0, 1]
            three features: [0, 1, 2]
        ----------------------------------
        the initial parameter need to computer:
            priori probability: self.p_priori
            conditional probability: self.p_condition

        :param train_data:
        :param train_label:
        :param test_data:
        :param test_label:
        """

        # test and train dataset
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label

        self.test_y_pred = None
        self.train_y_pred = None

        # Number of Y_categories
        categories = list(set(train_label))
        self.Y_num = len(categories)

        # Number of X_features
        self.X_features = np.zeros(self.train_data.shape[1])
        for i in range(self.train_data.shape[1]):
            feature_num = len(set(self.train_data[:, i]))
            self.X_features[i] = feature_num

        # initial priori probability
        self.p_priori = np.zeros(self.Y_num)

        # initial conditional probability
        # every feature has condition probability table
        # due to difference of feature, the size of table is different
        self.p_condition = list()
        for feature in range(self.train_data.shape[1]):
            p_condition_table = np.zeros((int(self.X_features[i]), self.Y_num))
            self.p_condition.append(p_condition_table)

    def train(self) -> None:
        """
        compute parameter:
            priori probability: self.p_priori
            conditional probability: self.p_condition
        :return:
        """

        # number of sample of train dataset
        sample_num = self.train_data.shape[0]

        # number of feature of one sample
        feature_num = self.train_data.shape[1]

        # initial number of y1...yn
        matrix_num = copy.deepcopy(self.p_condition)

        # the total number of events yi occurred
        # + self.Y_num: Laplace smoothing
        y_num = np.zeros(self.Y_num)

        # compute number of y_i in different events
        for i in range(sample_num):
            label = self.train_label[i]
            data = self.train_data[i]

            # compute total number of y_i
            self.p_priori[label] += 1

            # compute y_i conditional number
            for feature in range(0, feature_num):
                matrix_num[feature][data[feature], label] += 1

            # compute total number of y_i
            y_num[label] += 1

        # compute y_i priori probability
        self.p_priori = self.p_priori / sample_num

        # compute y_i condition probability
        for feature in range(feature_num):
            for y_i in range(self.Y_num):
                self.p_condition[feature][:, y_i] = matrix_num[feature][:, y_i] / y_num[y_i]

        self.train_y_pred = self.classify(self.train_data)[:, 1]

    def classify_one(self, data: list, out: bool = False) -> [float, int]:
        """
        return single result of classification
        :param out:
        :param data: the data need classify
        :return:
        """
        p = np.zeros(self.Y_num)
        for i in range(self.Y_num):
            p[i] = self.p_priori[i]
            for feature in range(len(data)):
                p[i] = p[i] * self.p_condition[feature][data[feature], i]

        value = max(p)
        idx = int(np.argmax(p, axis=0))

        if out:
            outfmt = 'probability of class {}: {:.4f}'
            for i in range(self.Y_num):
                print(outfmt.format(i, p[i]))

        return value, idx

    def classify(self, data: list) -> list:
        """
        return a group of data result of classification
        :param data:
        :return:
        """
        sample_num = data.shape[0]
        res = np.zeros((sample_num, 2))
        for i in range(sample_num):
            res[i, 0], res[i, 1] = self.classify_one(data[i, :])

        return res

    def test(self):
        """
        classify test dataset
        :return:
        """
        if not self.test_data:
            raise AttributeError('the test dataset has not load')

        self.test_y_pred = self.classify(self.test_data)

    @staticmethod
    def evaluation(y_test, y_pred):
        evaluator = Evaluator(y_test, y_pred)
        print(evaluator.classify_evaluation())


if __name__ == '__main__':

    datapath = 'D:\\workplace\\ML\\ML_tensorflow\\Naive_Bayes_Classifier\\datasample\\mail.xlsx'
    df = pd.read_excel(datapath)
    category = df.columns
    X = df.drop([category[0], category[-1]], axis=1)
    X = np.array(X)
    y = np.array(df[category[-1]])

    a = NaiveBayesClassifier(X, y)
    a.train()
    a.classify_one([0, 1, 1, 0])
    a.evaluation(a.train_label, a.train_y_pred)







