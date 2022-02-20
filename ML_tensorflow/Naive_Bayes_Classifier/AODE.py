"""
Semi-Naive Bayes Classifier
AODE method
"""

import numpy as np

from ML_tensorflow.Naive_Bayes_Classifier.NBC import NBC


class AODE(NBC):
    """
    AODE: Semi-Naive Bayes Classifier
    """

    def __init__(self, train_data, train_label, test_data=None, test_label=None, dis_index=None, cont_index=None):
        """
        initialize parameter:
            y_i_x_i: the number of x_i feature while label is y_i

        :param train_data:
        :param train_label:
        :param test_data:
        :param test_label:
        """
        super(AODE, self).__init__(train_data, train_label, test_data, test_label)

        # initialize y_i_x_i
        self.y_i_x_i = None

    def priori_prob(self) -> None:
        """
        compute priori probability
        P(y_i|x_i)
        -----------------------------
        1.record number and index of y_i
        2.record number of y_i and the feature x is xi
        -----------------------------
        :return: None
        """

        self.y_i_idx = list()

        # record number of x_i feature while label is y_i
        self.y_i_x_i = np.zeros((self.Y_num,
                                 self.train_data.shape[1],
                                 int(max(self.X_features))))

        for i in range(self.Y_num):
            self.y_i_idx.append([])

        for i in range(len(self.train_data)):
            label = self.train_label[i]
            feature = self.train_data[i]

            for j in range(len(feature)):
                self.y_i_x_i[label, j, feature[j]] += 1

            self.y_i_idx[label].append(i)

    def cond_prob_xi_xj(self, index_i: int, x_i: int, index_j: int, x_j: int, y_i: int) -> float:
        """
        compute conditional probability
        # P(xj|c, xi)
        :return: float
        """
        num_xj_yi_xi = 0
        for i in range(len(self.y_i_idx[y_i])):
            sample = self.train_data[self.y_i_idx[y_i][i]]

            if sample[index_i] == x_i and sample[index_j] == x_j:
                num_xj_yi_xi += 1
        prob = (num_xj_yi_xi + 1) / \
               (self.y_i_x_i[y_i, index_i, x_i] + self.X_features[index_j])

        return prob

    def posterior_prob(self, data: list, y_i: int) -> float:
        """
        compute posterior probability
        P(yi|X)
        :return:
        """

        prob = 0
        for i in range(len(data)):
            p = (self.y_i_x_i[y_i, i, data[i]] + 1) / \
                (self.train_data.shape[0] + self.X_features[i])
            p1 = 1
            for j in range(len(data)):
                p1 *= self.cond_prob_xi_xj(i, data[i], j, data[j], y_i)
            # P(y_i|x) = sum(P(y_i) * P(xi|y_i, xj))
            prob += p * p1

        return prob





