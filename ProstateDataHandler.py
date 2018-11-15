import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from categoric2numeric import *


class ProstateData:
    """
    Class to handle the Cite-U-Like data

    Predict:
    match_status

    """

    def __init__(self):
        self.file = './Data/Prostate.xlsx'
        self.sheet = 'Sheet1'
        self.raw_data = self.load()
        print(self.file)

    def load(self):
        data = pd.read_excel(self.file, sheet_name=self.sheet)
        return data

    def get_rawData(self):
        data = self.raw_data
        del data['ID']
        del data['train']
        return data

    def get_attributeNames(self):
        data = self.raw_data
        attributeNames = list(data.columns.values)
        return attributeNames

    def get_ClassificationFeatureData(self):
        data = self.raw_data
        X = data.values
        X_orig = np.copy(X)
        gleason = X_orig[:, 6]
        svi = X_orig[:, 4]
        X = np.delete(X, 6, 1)  # Deletes Gleason
        X = np.delete(X, 6, 1)  # Deletes PGG
        X = np.delete(X, 4, 1)  # Deletes SVI
        X_z = zscore(X)
        [X_Gleason, attribute_names_Gleason] = categoric2numeric(gleason)
        svi = np.reshape(svi, [97, 1])
        X_k = np.concatenate((X_z, X_Gleason, svi), axis=1)
        X_classification = X_k[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        y_classification = X_k[:, 10]
        N, M = X_classification.shape
        return N, M, X_classification, y_classification


if __name__ == '__main__':
    myData = ProstateData()
    raw_data = myData.get_rawData()
    attributeNames = myData.get_attributeNames()
    N, M, X, y = myData.get_ClassificationFeatureData()
    print(raw_data.head(5))
    print(attributeNames)
    print(N)
    print(M)

