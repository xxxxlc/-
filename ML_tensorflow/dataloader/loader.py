"""
read data and write inputs of tensorflow
"""

import tensorflow
import pandas as pd
import numpy as np
import random

from tensorflow import keras
from functools import singledispatch
from ML_tensorflow.dataloader.reader import FileReader


class Loader(object):
    """
    write inputs of model in tensorflow
    mainly build train and test dataset
    """

    def __init__(self) -> None:
        self.file_reader = FileReader()

        self.data = None
        self.label = None
        self.data_shape = None
        self.label_shape = None

        # Loader need divide data into two parts: train and test
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

    def readfile(self, filename: str) -> list:
        """
        use FileReader to load data
        :param filename:
        :return: np.array
        """
        file_data = self.file_reader.read(filename)
        return np.array(file_data)

    def data_loader(self, path: str) -> None:
        """
        load data for self.data
        :param path:
        :return:
        """
        self.data = self.readfile(path)
        self.data_shape = self.data.shape

        print("============================================")
        print("data already load: {}".format(self.data_shape))

    def label_loader(self, path):
        """
        load data for label
        if label and data in one file, you can rewrite this function
        :param path:
        :return:
        """
        self.label = self.readfile(path)
        self.label_shape = self.label.shape

        print("============================================")
        print("label already load: {}".format(self.data_shape))

    def data_divide(self, train_ratio: float = 0.8) -> None:
        """
        according to train_ratio, divide train and test dataset
        :param train_ratio:
        :return:
        """
        if self.data:
            self.train_data = self.data[0:self.data_shape[0] * train_ratio]
            self.test_data = self.data[self.data_shape[0] * train_ratio:-1]

        if self.label:
            self.train_label = self.label[0:self.label_shape[0] * train_ratio]
            self.test_label = self.label[self.label_shape[0] * train_ratio:-1]

    @staticmethod
    def shuffle(data: list, label: list) -> tuple:
        """
        shuffle data and label
        :param data:
        :param label:
        :return:
        """
        temp = zip(data, label)
        random.shuffle(temp)
        data, label = zip(*temp)
        return data, label

    def discrete_processor(self, data, features=None):
        """
        normalize discrete Data
        :param data:
        :param features:
        :return:
        """
        if not features:
            dis, con = self.type_data(data)
            features = dis

        sample = len(data)
        name_dic = list()

        if len(data.shape) == 1:
            res = np.zeros(sample)
            category = list(set(data))
            name_dic.append(category)
            for j in range(sample):
                res[j] = category.index(data[j])

            return res, name_dic

        for i in features:
            category = list(set(data[:, i]))
            name_dic.append(category)

            for j in range(sample):
                data[j, i] = category.index(data[j, i])

        return data, name_dic

    @staticmethod
    def continuous_processor():
        pass

    @staticmethod
    def type_data(data):
        """
        determine whether the datatype is discrete or continuous
        :param data: array
        :return: (list, list)
        """
        features = len(data[0])
        discrete = []
        continuous = []

        for i in range(features):
            sample = data[0]
            if isinstance(sample[i], float):
                continuous.append(i)
            else:
                discrete.append(i)

        return discrete, continuous

    @staticmethod
    def cleaner(data, trash=[None]):
        """
        return index of data which need remove
        :param trash: need to remove
        :param data:
        :return:
        """
        pass


if __name__ == "__main__":
    # path = 'D:\\workplace\\ML\\ML_tensorflow\\dataloader\\datasample\\A_pre.csv'
    filepath = 'D:\\workplace\\Group-of-ML\\learningNote\\Example\\chest-xray-pneumonia\\data_files_modify\\train\\NORMAL'
    a = Loader()
    a.data_loader(filepath)



