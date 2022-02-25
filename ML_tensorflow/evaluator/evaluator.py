"""
build evaluator to evaluate result of machine learning
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from ML_tensorflow.plot.plot import Plot


class Evaluator(object):

    def __init__(self, y_test, y_pred, y_score):
        self.y_test = y_test
        self.y_pred = y_pred

        # output probability
        self.y_score = y_score

    def classify_evaluation(self):
        """
        classify evaluator:
            classify accuracy,
            confusion matrix,
        :return: str
        """

        res = '----------------------\n'
        res += self.accuracy(self.y_test, self.y_pred)
        res += self.precision_recall_f1(self.y_test, self.y_pred)
        res += self.cfs_matrix(self.y_test, self.y_pred)

        self.pr(self.y_test, self.y_score, 0)

        return res

    @staticmethod
    def accuracy(y_test, y_pred):
        """
        return accuracy of y_test and y_pred
        :param y_test:
        :param y_pred:
        :return:
        """
        return 'Model accuracy score: {0:0.4f} \n'.format(accuracy_score(y_test, y_pred))

    @staticmethod
    def cfs_matrix(y_true, y_pred):
        """
        return confusion matrix
        :param y_true:
        :param y_pred:
        :return:
        """
        cm = confusion_matrix(y_true, y_pred)
        ans = 'confusion_matrix as follow: \n'

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        pic = Plot()
        # pic.plot_cfs_matrix(cm, None, [1, 2])
        pic.plot_cfs_matrix_sns(cm, None, [1, 2])

        for line in cm:
            ans += ' '.join((map(str, line)))
            ans += '\n'

        ans += 'confusion_matrix_normalized as follow: \n'

        for line in cm_normalized:
            ans += ' '.join((map(str, line)))
            ans += '\n'

        return ans

    @staticmethod
    def precision_recall_f1(y_true, y_pred, method='weighted'):
        """
        compute:
            1. precision
            2. recall
            3. f1 score
        :param y_pred:
        :param y_true:
        :param method: Weighting method
            1. weighted
            2. macro
            3. micro
        :return:
        """

        res = 'Weighted precision: {0:0.4f} \n'.format(precision_score(y_true, y_pred, average=method))
        res += 'Weighted recall: {0:0.4f} \n'.format(recall_score(y_true, y_pred, average=method))
        res += 'Weighted f1-score: {0:0.4f} \n'.format(f1_score(y_true, y_pred, average=method))

        return res

    @staticmethod
    def pr(y_true, y_score, y_c):
        """
        According to P-R curve to fixed data
        :param y_true: test labels
        :param y_score: output probability
        :param y_c: choose label in Multi-classification problem
        :return: a picture
        """
        y_score_idx = np.zeros(y_true.shape)
        y_true_pr = np.zeros(y_true.shape)

        for i in range(len(y_true)):
            y_score_idx[i] = y_score[i, y_true[i]]

            if y_true[i] == y_c:
                y_true_pr[i] = 1
            else:
                y_true_pr[i] = 0

        y_true_pr, y_score_idx = zip(*sorted(zip(y_true_pr, y_score_idx), key=lambda x: x[1], reverse=True))

        pic = Plot()
        pic.plot_pr(y_true_pr, y_score_idx)





