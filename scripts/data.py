'''
Data Class
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, annotation, label2predict = None, label2balance = None, fraction = None):
        self.annotation = annotation
        self.label2predict = label2predict
        self.fraction = fraction
        self.label2balance = label2balance
        self.training_data = None
        self.testing_data = None
    
    def balance_data(self, data, label):
        mincells = data[label].value_counts().min()
        ix = []
        for x in pd.unique(data[label]):
            ix.extend(data[data[label] == x].sample(n=mincells).index)
        data = data.loc[ix].copy()
        return data

    def split_data(self):
        self.training_data, self.testing_data = train_test_split(self.annotation, test_size = self.fraction, stratify = self.annotation[self.label2predict])
        #print(self.training_data[self.label2predict].value_counts())
        #print(self.testing_data[self.label2predict].value_counts())
        return self.training_data, self.testing_data

    def set_label(self):
        ord_enc = OrdinalEncoder()
        self.annotation[self.label2predict] = ord_enc.fit_transform(self.annotation[[self.label2predict]])
        return

    '''
    x: numpy array
    y: numpy array
    Cell_id: list
    num_labels: integer
    '''
    def make_data(self, typeOfData):
        data = None
        if typeOfData == "training":
            data = self.training_data
        elif typeOfData == "testing":
            data = self.testing_data

        cell_id = data.index.to_list()
        x = data.iloc[:, 0:len(data.columns)-1].copy()
        x = x.to_numpy()
        y = data[self.label2predict].copy()
        num_labels = len(set(y.to_list()))
        y = to_categorical(y, num_classes=num_labels, dtype='float32')
        return x, y, cell_id, num_labels

    '''
    return instances name to be removed from training data
    '''
    def make_LUFData(self, size):
        LUFData = self.training_data.sample(n = size)
        return LUFData.index.to_list()
        