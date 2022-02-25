"""
Base Classifier of Naive Bayes Classifier
"""

import numpy as np
import math

from ML_tensorflow.trainer.trainer import Trainer
from ML_tensorflow.evaluator.evaluator import Evaluator
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


class NBC(Trainer):
    """
    Base Classifier of Naive Bayes Classifier
    """

    def __init__(self, train_data, train_label, test_data=None, test_label=None, dis_index=None, cont_index=None):
        """
        build train and test dataset
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
        ----------------------------------------------
        sklearn:
        1. Multinomial Naive Bayes Classifier
        2. Bernoulli Naive Bayes Classifier
        3. Gaussian Naive Bayes Classifier

        :param train_data:
        :param train_label:
        :param test_data:
        :param test_label:
        """
        super(NBC, self).__init__(train_data, train_label, test_data, test_label)

        # discrete and continuous variables
        self.dis_index = dis_index
        self.cont_index = cont_index

        # record number of yi
        self.Y_num = len(set(self.train_label))

        # record index of  label is y_i
        # record number of each label
        # initialize priori probability
        self.y_i_idx = None
        self.y_i_num = None
        self.p_priori = None

        # record number of each feature
        self.X_features = np.zeros(self.train_data.shape[1])
        for i in range(self.train_data.shape[1]):
            feature_num = len(set(self.train_data[:, i]))
            self.X_features[i] = feature_num

    def priori_prob(self):
        """
        compute priori probability
        P(y_i)
        :return:
        """
        # initialize counter of yi idx
        self.y_i_idx = list()

        for i in range(self.Y_num):
            self.y_i_idx.append([])

        # initialize counter of yi num
        self.y_i_num = np.zeros(self.Y_num)

        # initialize priori probability
        self.p_priori = np.zeros(self.Y_num)

        for i in range(self.train_data.shape[0]):
            label = int(self.train_label[i])

            self.y_i_num[label] += 1
            self.y_i_idx[label].append(i)

        # Laplace_smoothing:
        # priori probability: P(y_i) = (N_i + 1) / (N_i + K)
        self.p_priori = (self.y_i_num + 1) / (self.train_data.shape[0] + self.Y_num)

    def cond_prob_dis(self, x_i, index, y_i):
        """
        discrete variables
        compute condition probability
        :return:
        """
        num_xi = 0
        for idx in self.y_i_idx[y_i]:
            if self.train_data[idx, index] == x_i:
                num_xi += 1
        prob = (num_xi + 1) / \
               (self.y_i_num[y_i] + self.X_features[index])
        return prob

    def cond_prod_cont(self, x_i, index, y_i):
        """
        continuous variables
        compute condition probability
        :return:
        """
        sample = self.train_data[self.y_i_idx[y_i], index]
        mu = np.average(sample)
        sigma = np.std(sample)

        prob = (1 / (math.sqrt(2 * math.pi)) *
                math.exp(-((x_i - mu) ** 2) / sigma))

        return prob

    def posterior_prob(self, data, y_i):
        """
        compute posterior probability
        P(y_i|X)
        :param data:
        :param y_i:
        :return:
        """

        prob = self.p_priori[y_i]
        for i in self.dis_index:
            prob *= self.cond_prob_dis(data[i], i, y_i)
        for i in self.cont_index:
            prob *= self.cond_prod_cont(data[i], i, y_i)

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

        if self.y_i_idx is None:
            self.priori_prob()

        # record posterior probability of each y_i
        p_y_x = np.zeros(self.Y_num)
        for y_i in range(self.Y_num):
            # compute P(y_i|c) posterior probability
            p_y_x[y_i] = self.posterior_prob(data, y_i)

        return p_y_x[np.argmax(p_y_x)], np.argmax(p_y_x), p_y_x

    def classify(self, data: list) -> [list, list]:
        """
        return a group of data result of classification
        :param data:
        :return:
        """
        sample_num = data.shape[0]
        res = np.zeros((sample_num, 2))

        # record all posterior probability
        y_score = np.zeros((sample_num, self.Y_num))

        for i in range(sample_num):
            res[i, 0], res[i, 1], y_score[i, :] = self.argmax_p(data[i, :])

        return res, y_score

    def test(self):
        """
        classify test dataset
        :return:
        """
        if not self.test_data:
            raise AttributeError('the test dataset has not load')

        self.test_y_pred = self.classify(self.test_data)

    def train_skl(self, function_type='GNB'):
        if function_type == 'MNB':
            trainer = MultinomialNB()
        elif function_type == 'BNB':
            trainer = BernoulliNB()
        else:
            trainer = GaussianNB()

        classifier = trainer.fit(self.train_data, self.train_label)
        if self.test_data:
            self.test_y_pred = classifier.predict(self.test_data)
        self.train_y_pred = classifier.predict(self.train_data)

        print('--------------------')
        print('sklearn have completed training')

    @staticmethod
    def dct_or_cnt(data: list) -> bool:
        """
        return feature is discrete or continuous
        :param data:
        :return: bool
        """
        category = set(data)
        for num in category:
            if int(num) != num:
                return False
        return True

    @staticmethod
    def evaluation(y_test, y_pred, y_score):
        """
        output evaluation of classifier
        :param y_score: output probability
        :param y_test: ture label
        :param y_pred: predict label
        :return:
        """
        evaluator = Evaluator(y_test, y_pred, y_score)
        print(evaluator.classify_evaluation())
