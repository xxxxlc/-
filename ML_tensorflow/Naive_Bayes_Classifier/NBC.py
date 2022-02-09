"""
Base Classifier of Naive Bayes Classifier
the aim is that reduce overlap code
"""

import numpy as np

from ML_tensorflow.trainer.trainer import Trainer
from ML_tensorflow.evaluator.evaluator import Evaluator


class NBC(Trainer):
    """
    Base Classifier of Naive Bayes Classifier
    """
    def __init__(self, train_data, train_label, test_data=None, test_label=None):
        """
        initial function
        :param train_data:
        :param train_label:
        :param test_data:
        :param test_label:
        """
        super(NBC, self).__init__(train_data, train_label, test_data, test_label)

        # record number of yi
        self.Y_num = len(set(self.train_label))

        # record number of each label
        # initialize priori probability
        self.y_i_num = None
        self.p_priori = None

        # record number of each feature
        self.X_features = np.zeros(self.train_data.shape[1])
        for i in range(self.train_data.shape[1]):
            feature_num = len(set(self.train_data[:, i]))
            self.X_features[i] = feature_num

        # initialize conditional probability
        self.p_condition = None

    def classify(self, data: list) -> list:
        """
        return a group of data result of classification
        :param data:
        :return:
        """
        return None

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


