"""
Semi-Naive Bayes Classifier
TAN method
"""
import math

import numpy as np

from ML_tensorflow.Naive_Bayes_Classifier.NBC import NBC


class TAN(NBC):
    """
    Semi-Naive Bayes Classifier
    Based on the maximum number of weighted generation tree
    """

    def __init__(self, train_data, train_label, test_data=None, test_label=None, dis_index=None, cont_index=None):
        """
        * number of features must >= 2
        :param train_data: 
        :param train_label: 
        :param test_data: 
        :param test_label: 
        :param dis_index: 
        :param cont_index: 
        """
        super(TAN, self).__init__(train_data, train_label, test_data, test_label)

        # weight matrix which used to save I(xi, xj|y)
        self.edge = None

        # maximum spanning tree
        # bool
        # set make edge directed
        self.graph = None

    def argmax_p(self, data):
        """
        compute all P(y|x)
        select max P(y|x)
        other task:
            1. compute priori probability
            2. build maximum spanning tree
        :param data:
        :return:
        """
        if len(data) != len(self.X_features):
            raise AttributeError('number of features is error')

        if self.y_i_idx is None:
            self.priori_prob()

        if self.edge is None:
            self.construct_weight_map()

        if self.graph is None:
            self.prim_maximum_spanning_tree()

        # record posterior probability of each y_i
        p_y_x = np.zeros(self.Y_num)
        for y_i in range(self.Y_num):
            # compute P(y_i|c) posterior probability
            p_y_x[y_i] = self.posterior_prob(data, y_i)

        return p_y_x[np.argmax(p_y_x)], np.argmax(p_y_x)

    def posterior_prob(self, data, y_c):
        """
        compute posterior probability
        P(yi|X) = P(yi) * prod(P(xi|yi,pai))
        :param data:
        :param y_c:
        :return:
        """

        # priori probability
        p = self.p_priori[y_c]

        # condition probability
        p1 = 1
        for i in range(len(data)):
            for j in range(len(data)):
                if self.graph[i][j] == 1:
                    p1 *= self.cond_prob_xi_xj(i, data[i], j, data[j], y_c)

        return p * p1

    def cond_prob_xi_xj(self, index_i, x_i, index_j, x_j, y_c):
        """
        compute conditional probability
        # P(xj|c, xi)
        :return: float
        """
        num_xj_yc_xi = 0
        num_xi_yc = 0
        for i in range(len(self.y_i_idx[y_c])):
            sample = self.train_data[self.y_i_idx[y_c][i]]

            if sample[index_i] == x_i and sample[index_j] == x_j:
                num_xj_yc_xi += 1
            if sample[index_i] == x_i:
                num_xi_yc += 1
        prob = (num_xj_yc_xi + 1) / \
               (num_xi_yc + self.X_features[index_j])

        return prob

    def mutual_info(self, index_i, index_j):
        """
        I(xi, xj|y) = sum P(xi, xj|yc) * log(P(xi, xj|yc) / (P(xj|yc) * P(xj|yc)))
        :param index_i:
        :param index_j:
        :return:
        """

        I_xi_xj = 0
        for y_c in range(self.Y_num):
            # num_xi_xj_yc = 0
            # num_xi_yc = 0
            # num_xj_yc = 0

            for x_i in range(int(self.X_features[index_i])):
                for x_j in range(int(self.X_features[index_j])):
                    num_xi_xj_yc = 0
                    num_xi_yc = 0
                    num_xj_yc = 0

                    for i in range(len(self.y_i_idx[y_c])):
                        sample = self.train_data[self.y_i_idx[y_c][i]]

                        if sample[index_i] == x_i:
                            num_xi_yc += 1
                        if sample[index_j] == x_j:
                            num_xj_yc += 1
                        if sample[index_i] == x_i and sample[index_j] == x_j:
                            num_xi_xj_yc += 1

                    # P(xi,xj|yc) = (D_yc_xi_xj + 1) / (D_yc + N_yc)
                    p_xi_xj_yc = (num_xi_xj_yc + 1) / (len(self.y_i_idx[y_c]) + self.Y_num)

                    # P(xi|yc) = (D_yc_xi + 1) / (D_yc + N_xi)
                    p_xi_yc = (num_xi_yc + 1) / (len(self.y_i_idx[y_c]) + self.X_features[index_i])

                    # P(xj|yc) = (D_yc_xj + 1) / (D_yc + N_xj)
                    p_xj_yc = (num_xj_yc + 1) / (len(self.y_i_idx[y_c]) + self.X_features[index_j])

                    I_xi_xj += p_xi_xj_yc * math.log10(p_xi_xj_yc / (p_xj_yc * p_xi_yc))

        return I_xi_xj

    def construct_weight_map(self):
        """
        compute weight matrix
        :return:
        """

        feature_num = len(self.train_data[0])

        self.edge = np.zeros((feature_num, feature_num))

        for i in range(0, feature_num):
            for j in range(i + 1, feature_num):
                self.edge[i, j] = self.mutual_info(i, j)
                self.edge[j, i] = self.edge[i, j]

    def prim_maximum_spanning_tree(self):
        """
        prim algorithm to build maximum spanning tree
        self.graph
        :return:
        """

        feature_num = len(self.train_data[0])

        if feature_num < 2:
            raise AttributeError('number of features must > 2')

        self.graph = np.zeros((feature_num, feature_num))

        # maximum spanning tree
        span_tree = list()

        all_node = set(range(0, feature_num))

        # pick the first attribute (index == 0)
        # selected node
        selected_node = set()
        selected_node.add(0)

        # other node which not be selected
        # not selected node
        candidate_node = set(range(1, feature_num))

        while selected_node != all_node:
            max_weight = float('-inf')
            select = None
            candidate = None

            for i in selected_node:
                for j in candidate_node:
                    if self.edge[i, j] > max_weight:
                        select = i
                        candidate = j
                        max_weight = self.edge[i, j]

            span_tree.append([select, candidate, max_weight])
            selected_node.add(candidate)
            candidate_node.remove(candidate)

        # turn max edge to 1
        for i in range(len(span_tree)):
            self.graph[span_tree[i][0]][span_tree[i][1]] = 1











