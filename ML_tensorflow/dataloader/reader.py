"""
read data in different type files
"""
import os
import pandas as pd
import numpy as np

from PIL import Image


class FileReader(object):
    """
    read data come from different types' files
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def read_file(filepath: str) -> list:
        """
        read data in single file: csv, xlsx, txt, jpg, png, jpeg
        :param filepath:
            a filepath
        :return: list
            data in file
        """
        picture_type = ('jpg', 'png', 'jpeg')
        folder_path, filetype = filepath.split('.')
        if filetype == 'csv':
            data = pd.read_csv(filepath)
            data = data.values
        elif filetype == 'xlsx':
            # if data in excel is complex, (such as many sheet in excel)
            # write new code to adapt data in excel
            data = pd.read_excel(filepath)
            data = data.values
        elif filetype in picture_type:
            data = np.array(Image.open(filepath))
        else:
            pass

        return data

    def read_folder(self, folder: str) -> list:
        """
        read data in a folder: csv, xlsx, txt, jpg, png
        :param folder:
            path of folder which need read
        :return: list
            list of data in single file
        """
        all_file_list = os.listdir(folder)
        folder_data = []
        for single_file in all_file_list:
            single_file_data = self.read_file(folder + '\\' + single_file)
            folder_data.append(single_file_data)

        return folder_data

    def read(self, path: str) -> list:
        """
        determine read file or folder
        :param path:
        :return: the function which read file or folder
        """
        if not os.path.exists(path):
            raise AttributeError('this path do not exist')
        if os.path.isdir(path):
            return self.read_folder(path)
        if os.path.isfile(path):
            return self.read_file(path)

    def picture_modify(self, img) -> list:
        """
        modify image which read from file
        :param img:
        :return:
        """
        pass


if __name__ == "__main__":
    pass
