"""
build evaluator to evaluate result of machine learning
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class Evaluator(object):

    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred

    def classify_evaluation(self):
        """
        classify evaluator:
            classify accuracy,
            confusion matrix,
        :return: str
        """

        res = '----------------------\n'
        res += self.accuracy(self.y_test, self.y_pred)
        res += self.cfs_matrix(self.y_test, self.y_pred)

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
    def cfs_matrix(y_test, y_pred):
        """
        return confusion matrix
        :param y_test:
        :param y_pred:
        :return:
        """
        cm = confusion_matrix(y_test, y_pred)
        ans = 'confusion_matrix as follow: \n'

        for line in cm:
            ans += ' '.join((map(str, line)))
            ans += '\n'

        return ans


