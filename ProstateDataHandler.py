import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from categoric2numeric import *
from similarity import binarize2


class ProstateData:
    """
    Class to handle the Cite-U-Like data

    Predict:
    match_status

    """

    def __init__(self):
        print('DataHandler for Prostate data initialized')
        self.file = './Data/Prostate.xlsx'
        self.sheet = 'Sheet1'
        self.raw_data = self.load()

    def load(self):
        data = pd.read_excel(self.file, sheet_name=self.sheet)
        return data

    def get_rawData(self):
        data = self.raw_data
        del data['ID']
        del data['train']
        return data

    def get_attributeNames(self):
        attributeNames = ['lCaVol', 'lWeight', 'Age', 'lBPH', 'lCP', 'lPSA', 'Gleason 6.0', 'Gleason 7.0', 'Gleason 8.0', 'Gleason 9.0']
        print('Attribute Names in data set are: {}'.format(attributeNames))
        return attributeNames

    def get_classLabels(self):
        data = self.raw_data
        classLabels = data['SVI'].values.tolist()
        classNames = sorted(set(classLabels))
        classDict = dict(zip(classNames, range(4)))
        return classLabels, classNames, classDict

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
        X_k = np.concatenate((X_z, X_Gleason), axis=1)
        X_classification = X_k#[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        y_classification = svi.squeeze() #X_k[:, 10]
        N, M = X_classification.shape
        print('There are {} features in the data set'.format(M))
        print('There are {} observations in the data set'.format(N))
        print('X has shape {}'.format(np.shape(X_classification)))
        print('y has shape {}'.format(np.shape(y_classification)))
        
        return N, M, X_classification, y_classification, svi
    
    def get_OutlierFeatureData(self):
        data = self.raw_data
        X = data.values
        X_orig = np.copy(X)
        gleason = X_orig[:, 6]
        svi = X_orig[:, 4]
        X = np.delete(X, 6, 1)  # Deletes Gleason
        X = np.delete(X, 6, 1)  # Deletes PGG
        #X = np.delete(X, 4, 1)  # Deletes SVI
        X_z = zscore(X)
        [X_Gleason, attribute_names_Gleason] = categoric2numeric(gleason)
        svi = np.reshape(svi, [97, 1])
        X_k = np.concatenate((X_z, X_Gleason), axis=1)
        N, M = X_classification.shape
        print('There are {} features in the data set'.format(M))
        print('There are {} observations in the data set'.format(N))
        print('X has shape {}'.format(np.shape(X_k)))
        
        return N, M, X_k

    def get_binarizedFeatureData(self):
        _, _, X, y, _ = self.get_ClassificationFeatureData()
        attributeNames = self.get_attributeNames()
        gleason = X[:, 6:]
        X = X[:,0:6]

        gleasonNames = attributeNames[6:]
        attributeNames = attributeNames[0:6]
        [y, _] = categoric2numeric(y)

        yNames = ['SVI 0', 'SVI 1']

        Xbin, attributeNamesBin = binarize2(X, attributeNames)
        Xbin = np.concatenate((Xbin, gleason), axis=1)

        Xbin = np.concatenate((Xbin, y), axis=1)
        attributeNamesBin = attributeNamesBin + gleasonNames + yNames
        print('X has now been transformed into:')
        print(Xbin)
        print(attributeNamesBin)
        return Xbin, attributeNamesBin

    

if __name__ == '__main__':
    myData = ProstateData()
    raw_data = myData.get_rawData()
    attributeNames = myData.get_attributeNames()
    classLabels, classNames, classDict = myData.get_classLabels()
    N, M, X, y = myData.get_ClassificationFeatureData()
    print(raw_data.head(5))
    print(attributeNames)
    print(N)
    print(M)

