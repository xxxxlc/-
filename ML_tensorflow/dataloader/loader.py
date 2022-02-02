"""
read data and write inputs of tensorflow
"""

import tensorflow
import pandas as pd
import numpy as np

from tensorflow import keras
from functools import singledispatch
from reader import FileReader


class Loader:
    """
    write inputs of model in tensorflow
    """

    def __init__(self) -> None:
        self.file_reader = FileReader()

        # Loader need divide data into two parts: train and test
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []

    def readfile(self, filename: str) -> list:
        file_data = self.file_reader.read(filename)
        return np.array(file_data)



if __name__ == "__main__":
    # path = 'D:\\workplace\\ML\\ML_tensorflow\\dataloader\\datasample\\A_pre.csv'
    path = 'D:\\workplace\\Group-of-ML\\learningNote\\Example\\chest-xray-pneumonia\\data_files_modify\\train\\NORMAL'
    a = Loader()
    a.readfile(path)


