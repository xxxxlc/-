"""
Watermelon dataset
"""


from ML_tensorflow.dataloader.loader import Loader
from ML_tensorflow.Naive_Bayes_Classifier.NaiveBayesClassifier import NaiveBayesClassifier
from ML_tensorflow.Naive_Bayes_Classifier.NBC import NBC

data_path = "D:\\workplace\\ML\\ML_tensorflow\\Naive_Bayes_Classifier\\datasample\\watermelon30.xlsx"
loader = Loader()
X, y = loader.file_reader.read_excel(data_path)
dis_index, con_index = loader.type_data(X)
X, X_name = loader.discrete_processor(X)
y, y_name = loader.discrete_processor(y)

a = NaiveBayesClassifier(X, y)
a.train()
a.evaluation(a.train_label, a.train_y_pred)

b = NBC(X, y, [], [], dis_index, con_index)
y_pred = b.classify(X)
b.evaluation(y, y_pred[:, 1])


