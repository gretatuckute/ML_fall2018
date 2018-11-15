import ProstateDataHandler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Nicer formatting of plots
sns.set_style("darkgrid")
sns.set_palette(sns.dark_palette("purple"))


print('Creating DataHandler object')
myData = ProstateDataHandler.ProstateData()

print('Importing raw data')
raw_data = myData.get_rawData()

print('Collecting Attribute names and class labels')
attributeNames = myData.get_attributeNames()

print('Attribute Names in data set are: {}'. format(attributeNames))

print('Preparing data for classification')
N, M, X, y = myData.get_ClassificationFeatureData()

print('There are {} features in the data set'.format(M))
print('There are {} observations in the data set'.format(N))
print('X has shape {}'.format(np.shape(X)))
print('y has shape {}'.format(np.shape(y)))



