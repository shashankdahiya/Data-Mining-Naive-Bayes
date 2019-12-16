import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from naivebayes import naivebayes


def accuracy(actual, predicted):
    accuracy = np.sum(actual == predicted) / len(actual)
    return accuracy
#t_data -> training sample & test_data -> testing sample 
t_data_before_split, t_label_before_split = datasets.make_classification(n_samples=1236, n_features=15, n_classes=2)#, random_state=210)
t_data, test_data, t_label, test_label = train_test_split(t_data_before_split, t_label_before_split, test_size=0.2) #, random_state=123)

nb = nb()
nb.fit(t_data, t_label)
predictions = nb.predict(test_data)

print("Naive Bayes classification accuracy", accuracy(test_label, predictions))