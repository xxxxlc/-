"""
read data and write inputs of tensorflow
"""

import tensorflow
import pandas as pd
import numpy as np

from tensorflow import keras
from functools import singledispatch


class FileReader:
    """
    read data come from different types' files
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def read_file(filepath: str) -> list:
        """
        read data in single file: csv, xlsx, txt
        :param filepath: str
        :return: data
        """
        folder_path, filetype = filepath.split('.')
        if filetype == 'csv':
            data = pd.read_csv(filepath)
            data = data.values
        elif filetype == 'xlsx':
            # if data in excel is complex, (such as many sheet in excel)
            # write new code to adapt data in excel
            data = pd.read_excel(filepath)
            data = data.values
        else:
            pass

        return data

    def read_folder(self, folder: str) -> list:
        pass


class Loader:
    """
    write inputs of model in tensorflow
    """
    def __init__(self) -> None:
        pass

    def readfile(self, filename: str) -> str:
        pass


if __name__ == "__main__":
    path = 'D:\workplace\ML\ML-tensorflow\dataloader\datasample\A_pre.csv'
    fileloader = FileReader()
    Data = fileloader.read_file(path)