"""
get the picture
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Plot(object):
    """
    all pictures are draw there
    """
    def __init__(self):
        pass

    @staticmethod
    def plot_cfs_matrix(cm, save_path, classes, title='Confusion Matrix'):
        """
        draw confusion matrix
        :param classes: list
        :param cm: list
            confusion matrix
        :param save_path: str
            the way save image
        :param title: str
        :return: None
        """

        plt.figure(figsize=(12, 8), dpi=100)

        # set precision of output is 2: decimal places is 2
        np.set_printoptions(precision=2)

        ind_array = np.arange(len(classes))
        # generate coordinate matrix
        x, y = np.meshgrid(ind_array, ind_array)

        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val, x_val]

            if c > 0.001:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

        plt.imshow(cm, interpolation='nearest', cmap='YlGnBu')
        plt.title(title)
        plt.colorbar()
        x_location = np.array(range(len(classes)))
        plt.xticks(x_location, classes, rotation=90)
        plt.yticks(x_location, classes)
        plt.ylabel('Actual label')
        plt.xlabel('Predict label')

        # offset the tick
        tick_marks = np.array(range(len(classes))) + 0.5
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)

        # show confusion matrix
        # plt.savefig(save_name, format='png')
        plt.show()

    @staticmethod
    def plot_cfs_matrix_sns(cm, save_path, classes, title='Confusion Matrix'):
        """
        use seaborn to draw confusion matrix
        :param cm:
        :param save_path:
        :param classes:
        :param title:
        :return:
        """

        conf_matrix = pd.DataFrame(cm, classes, classes)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()

