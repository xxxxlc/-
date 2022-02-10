"""
Semi-Naive Bayes Classifier
"""

import math
import numpy as np
import pandas as pd

from ML_tensorflow.Naive_Bayes_Classifier.NBC import NBC


class AODE(NBC):
    """
    AODE: Semi-Naive Bayes Classifier
    """

    def __init__(self, train_data, train_label, test_data=None, test_label=None):
        """

        :param train_data:
        :param train_label:
        :param test_data:
        :param test_label:
        """
        super(AODE, self).__init__(train_data, train_label, test_data, test_label)

        # record number of x_i feature while label is y_i
        self.y_i_x_i = np.zeros((self.Y_num,
                                 self.train_data.shape[1],
                                 int(max(self.X_features))))

        # record index of  label is y_i
        self.idx_yi = list()

        self.priori_probability()

    def priori_probability(self) -> None:
        """
        compute priori probability
        P(y_i|x_i)
        :return:
        """

        for i in range(self.Y_num):
            self.idx_yi.append([])

        for i in range(len(self.train_data)):
            label = self.train_label[i]
            feature = self.train_data[i]

            for j in range(len(feature)):
                self.y_i_x_i[label, j, feature[j]] += 1

            self.idx_yi[label].append(i)

    def condition_probability(self, index_i, x_i, index_j, x_j, y_i):
        """
        compute conditional probability
        # P(xj|c, xi)
        :return:
        """
        num_xj_yi_xi = 0
        for i in range(len(self.idx_yi[y_i])):
            sample = self.train_data[self.idx_yi[y_i][i]]

            if sample[index_i] == x_i and sample[index_j] == x_j:
                num_xj_yi_xi += 1
        prob = (num_xj_yi_xi + 1) / \
               (self.y_i_x_i[y_i, index_i, x_i] + self.X_features[index_j])

        return prob

    def posterior_probability(self, data, y_i):
        """
        compute posterior probability
        :return:
        """
        prob = 0
        for i in range(len(data)):
            p = (self.y_i_x_i[y_i, i, data[i]] + 1) / \
                (self.train_data.shape[0] + self.X_features[i])
            p1 = 1
            for j in range(len(data)):
                p1 *= self.condition_probability(i, data[i], j, data[j], y_i)
            # P(y_i|x) = sum(P(y_i) * P(xi|y_i, xj))
            prob += p * p1

        return prob

    def argmax_p(self, data):
        """
        compute all P(y|x)
        select max P(y|x)
        :param data:
        :return:
        """
        if len(data) != len(self.X_features):
            raise AttributeError('number of features is error')

        # record posterior probability of each y_i
        p_y_x = np.zeros(self.Y_num)
        for y_i in range(self.Y_num):
            # compute P(y_i|c) posterior probability
            p_y_x[y_i] = self.posterior_probability(data, y_i)

        return p_y_x[np.argmax(p_y_x)], np.argmax(p_y_x)

    def classify(self, data: list) -> list:
        """
        return a group of data result of classification
        :param data:
        :return:
        """
        sample_num = data.shape[0]
        res = np.zeros((sample_num, 2))
        for i in range(sample_num):
            res[i, 0], res[i, 1] = self.argmax_p(data[i, :])

        return res




