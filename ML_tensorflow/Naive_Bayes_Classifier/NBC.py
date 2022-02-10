"""
Base Classifier of Naive Bayes Classifier
the aim is that reduce overlap code
"""

import numpy as np

from ML_tensorflow.trainer.trainer import Trainer
from ML_tensorflow.evaluator.evaluator import Evaluator
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


class NBC(Trainer):
    """
    Base Classifier of Naive Bayes Classifier
    """
    def __init__(self, train_data, train_label, test_data=None, test_label=None):
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
        ----------------------------------
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

    def classify(self, data: list) -> None:
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
    def evaluation(y_test, y_pred):
        """
        output evaluation of classifier
        :param y_test: ture label
        :param y_pred: predict label
        :return:
        """
        evaluator = Evaluator(y_test, y_pred)
        print(evaluator.classify_evaluation())



