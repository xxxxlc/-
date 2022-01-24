"""
read data and write inputs of tensorflow
"""

import tensorflow
import pandas as pd
import numpy as np

from tensorflow import keras
from functools import singledispatch
from .reader import FileReader


class Loader:
    """
    write inputs of model in tensorflow
    """

    def __init__(self) -> None:
        self.filereader = FileReader()

    def readfile(self, filename: str) -> str:
        pass


if __name__ == "__main__":
    path = 'D:\workplace\ML\ML-tensorflow\dataloader\datasample\A_pre.csv'

