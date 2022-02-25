"""
this file solve titanic classify problem
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ML_tensorflow.dataloader.loader import Loader
from ML_tensorflow.Naive_Bayes_Classifier.NBC import NBC


if __name__ == '__main__':
    train_dataset_path = "/problem/datasample/Titanic/train.csv"
    test_dataset_path = "/problem/datasample/Titanic/test.csv"

    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)

    category = df_train.columns

    # train_labels = np.array(df_train['Survived'])
    # train_data = np.array(df_train.drop(category[1], axis=1))
    #
    # test_data = np.array(df_test)

    # there are lots of null in 'Cabin', so drop it
    df_train.drop('Cabin', axis=1, inplace=True)
    df_test.drop('Cabin', axis=1, inplace=True)

    # because feature 'Age' has some missing, use median age to replace them
    median_age = df_train['Age'].median()
    df_train['Age'] = df_train['Age'].fillna(median_age)

    median_age = df_test['Age'].median()
    df_test['Age'] = df_test['Age'].fillna(median_age)

    # for char column, get the most frequent of char for where is missing
    # freq_port: the most frequent of char
    freq_port = df_train.Embarked.dropna().mode()[0]
    # replace missing of 'Embarked' with the most frequent char
    df_train['Embarked'] = df_train['Embarked'].fillna(freq_port)

    freq_port = df_test.Embarked.dropna().mode()[0]
    df_test['Fare'] = df_test['Fare'].fillna(freq_port)

    # print(df_train.isnull().sum())
    # print(df_test.isnull().sum())


    load = Loader()

    train_label = df_train['Survived']

    considerate_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
    train_data = np.array(df_train[considerate_cols])
    test_data = np.array(df_test[considerate_cols])

    train_data, name_dic = load.discrete_processor(train_data, [0, 1])
    test_data, name_dic = load.discrete_processor(test_data, [0, 1])

    nbc_classifier = NBC(train_data, train_label, [], [], [0, 1], [2, 3, 4])
    y_pred, y_score = nbc_classifier.classify(test_data)

    result = list(y_pred[:, 1])
    serial = list(df_test['PassengerId'])

    df = pd.DataFrame({'PassengerId': serial, 'Survived': result})
    df.to_csv("result.csv", index=False, sep=',')









