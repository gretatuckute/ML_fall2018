import ProstateDataHandler
import numpy as np
import seaborn as sns
import report_PCA
import ML_plotter
from Clustering import HierarchicalCluster

# Nicer formatting of plots
sns.set_style("darkgrid")


print('Creating DataHandler object')
myData = ProstateDataHandler.ProstateData()

print('Importing raw data')
raw_data = myData.get_rawData()

print('Collecting Attribute names and class labels')
attributeNames = myData.get_attributeNames()
classLabels, classNames, classDict = myData.get_classLabels()
C = len(classNames)

print('Attribute Names in data set are: {}'. format(attributeNames))

print('Preparing data for classification')
N, M, X, y = myData.get_ClassificationFeatureData()

print('There are {} features in the data set'.format(M))
print('There are {} observations in the data set'.format(N))
print('X has shape {}'.format(np.shape(X)))
print('y has shape {}'.format(np.shape(y)))

'''
ML_plotter.plot_attributes_2d(i=0,
                              j=1,
                              X=X,
                              y=y,
                              C=C,
                              classNames=classNames,
                              attributeNames=attributeNames,
                              saveFigure=False)
'''

prostatePCA = report_PCA.prostatePCA(X=X)
X_pca = prostatePCA.get_principal_components(n_pca=2)
prostatePCA.display_pca()

prostate_hierarchical_clustering = HierarchicalCluster(X=X_pca, y=y, method='average')
prostate_hierarchical_clustering.display_cluster_plot(max_cluster=2)
prostate_hierarchical_clustering.display_dendogram(max_display_levels=100)
