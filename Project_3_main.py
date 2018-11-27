import ProstateDataHandler
import numpy as np
import seaborn as sns
import report_PCA
import ML_plotter
from Clustering import HierarchicalCluster
import Association

# Nicer formatting of plots
sns.set_style("darkgrid")

# Controllers
set_feature_plot = False
set_hierarchical_clustering = False
set_display_pca = False
set_association_mining = True


# Creating object to handle data load and feature transformation
myData = ProstateDataHandler.ProstateData()

raw_data = myData.get_rawData()


# Preparing attribute names
attributeNames = myData.get_attributeNames()
classLabels, classNames, classDict = myData.get_classLabels()
C = len(classNames)

# Generating X & Y and dimensionality
N, M, X, y = myData.get_ClassificationFeatureData()


# Plotting two features against each other
if set_feature_plot:
    ML_plotter.plot_attributes_2d(i=0,
                                j=1,
                                X=X,
                                y=y,
                                C=C,
                                classNames=classNames,
                                attributeNames=attributeNames,
                                saveFigure=False)


if set_hierarchical_clustering:
    # Creating a PCA object
    prostatePCA = report_PCA.prostatePCA(X=X)

    # Projecting the data down to the first two principal components (easier plotting)
    X_pca = prostatePCA.get_principal_components(n_pca=2)

    # Displaying the first two principal components against each other
    if set_display_pca:
        prostatePCA.display_pca()

    # Perform hierarchical clustering
    prostate_hierarchical_clustering = HierarchicalCluster(X=X, y=y, method='average')
    prostate_hierarchical_clustering.display_cluster_plot(max_cluster=2, centroids=True)
    prostate_hierarchical_clustering.display_dendogram(max_display_levels=100, orientation='top', color_threshold=3.2)
    prostate_hierarchical_clustering.validate_cluster(K=10)


if set_association_mining:
    Xbin, attributeNamesBin = myData.get_binarizedFeatureData()
    prostate_association_mining = Association.AssociationMining()
    transactions = prostate_association_mining.mat2transactions(Xbin, attributeNamesBin)
    rules = prostate_association_mining.get_rules(t=transactions, print_rules=True)

