"""
Naive Bayes Classifier
"""
import math

import numpy as np
import copy

from ML_tensorflow.Naive_Bayes_Classifier.NBC import NBC


class NaiveBayesClassifier(NBC):
    """
    Base Naive Bayes Classifier
    function:
        supported discrete feature variables
        supported continuous feature variables
        ----------------------------------------------
        sklearn:
        1. Multinomial Naive Bayes Classifier
        2. Bernoulli Naive Bayes Classifier
        3. Gaussian Naive Bayes Classifier
    """

    def __init__(self, train_data, train_label, test_data=None, test_label=None) -> None:
        """
        build train and test dataset,
        build priori probability,
        build conditional probability.
        ----------------------------------
        discrete feature variables:
            input label data must be list of integer:
            for example:
                two labels: [0, 1]
                three labels: [0, 1, 2]
            as same as feature:
                two features: [0, 1]
                three features: [0, 1, 2]
        continuous feature variables:
            input label data must be constructed in [0, 1]
        ----------------------------------
        the initial parameter need to computer:
            priori probability: self.p_priori
            conditional probability: self.p_condition

        :param train_data:
        :param train_label:
        :param test_data:
        :param test_label:
        """
        super(NaiveBayesClassifier, self).__init__(train_data, train_label, test_data, test_label)

        # separate discrete feature variables and continuous feature variables
        self.variable_type = np.zeros(self.train_data.shape[1])
        for feature in range(self.train_data.shape[1]):
            self.variable_type[feature] = self.dct_or_cnt(self.train_data[:, feature])

        # Number of X_features
        self.X_features = np.zeros(self.train_data.shape[1])
        for i in range(self.train_data.shape[1]):
            if self.variable_type[i]:
                feature_num = len(set(self.train_data[:, i]))
                self.X_features[i] = feature_num
            else:
                self.X_features[i] = 2

        # initial priori probability
        # self.y_num_priori record number of y_i
        # self.p_priori record priori probability of y_i
        self.y_num_priori = np.zeros(self.Y_num)
        self.p_priori = np.zeros(self.Y_num)

        # initial conditional probability
        # every feature has condition probability table
        # due to difference of feature, the size of table is different
        # self.num_condition record matrix of (xi | y_i)
        # self.p_condition record condition probability of (xi | y_i)
        self.num_condition = None
        self.p_condition = list()
        for feature in range(self.train_data.shape[1]):
            p_condition_table = np.zeros((int(self.X_features[feature]), self.Y_num))
            self.p_condition.append(p_condition_table)

        # initial variance and mean of continuous feature variables
        self.sigma = np.zeros((self.train_data.shape[1], self.Y_num))
        self.mu = np.zeros((self.train_data.shape[1], self.Y_num))

        # After laplace_smoothing
        # self.laplace_p_priori = np.zeros(self.Y_num)
        # self.laplace_p_condition = copy.deepcopy(self.p_condition)

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
        # ignore continuous feature variables, =0
        matrix_num = copy.deepcopy(self.p_condition)

        # separate y_i to compute variance and mean
        # compute continuous feature variables
        list_y_i = [[] for _ in range(self.Y_num)]

        # compute discrete variables number of y_i in different events
        for i in range(sample_num):
            label = self.train_label[i]
            data = self.train_data[i]

            # compute total number of y_i
            self.y_num_priori[label] += 1

            # compute y_i conditional number
            for feature in range(0, feature_num):
                # only discrete variables can be record number
                if self.variable_type[feature]:
                    matrix_num[feature][data[feature], label] += 1
                # continuous variables
                else:
                    # separate y_i
                    list_y_i[label].append(data[feature])

        # compute y_i priori probability
        self.p_priori = self.y_num_priori / sample_num

        # record discrete variables matrix of (xi | y_i)
        self.num_condition = matrix_num

        # compute discrete variables y_i condition probability
        for feature in range(feature_num):
            for y_i in range(self.Y_num):
                if self.variable_type[feature]:
                    self.p_condition[feature][:, y_i] = matrix_num[feature][:, y_i] / self.y_num_priori[y_i]
                else:
                    # compute variance and mean of continuous feature variables
                    array = np.array(list_y_i[feature])
                    self.mu[feature, y_i] = np.mean(array)
                    self.sigma[feature, y_i] = np.var(array)
                    self.p_condition[feature][0, y_i] = self.mu[feature, y_i]
                    self.p_condition[feature][1, y_i] = self.sigma[feature, y_i]

        self.train_y_pred = self.classify(self.train_data)[:, 1]

    def laplace_smoothing(self):
        """
        Laplace_smoothing:
            priori probability: P(y_i) = (N_i + 1) / (N_i + K)
            condition probability: P(x_i|y_i) = (N_ij + 1) / (N_i + S_j)
        :return:
        """
        sample_num = self.train_data.shape[0]
        self.p_priori = (self.y_num_priori + 1) / (sample_num + self.Y_num)

        feature_num = self.train_data.shape[1]
        for feature in range(feature_num):
            for y_i in range(self.Y_num):
                if self.variable_type[feature]:
                    self.p_condition[feature][:, y_i] = ((self.num_condition[feature][:, y_i] + 1) /
                                                         (self.y_num_priori[y_i] + len(self.num_condition[feature])))

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
                if self.variable_type[feature]:
                    p[i] = p[i] * self.p_condition[feature][data[feature], i]
                else:
                    mu = self.mu[feature, i]
                    sigma = self.sigma[feature, i]
                    p[i] = p[i] * (1 / (math.sqrt(2 * math.pi)) * math.exp(-((data[feature] - mu) ** 2) / sigma))

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

