"""
initial trainer
"""


class Trainer(object):
    """
    Base trainer
    """
    def __init__(self, train_data, train_label, test_data=None, test_label=None):
        """
        Initialize the basic properties of the trainer
        every trainer has its train and test dataset:
            self.train_data: ndarray
            self.train_label: ndarray
            self.test_data: ndarray
            self.test_label: ndarray
        :param train_data:
        :param train_label:
        :param test_data:
        :param test_label:
        """
        self.train_data = train_data
        self.train_label = train_label

        self.test_data = test_data
        self.test_label = test_label

        self.test_y_pred = None
        self.train_y_pred = None

