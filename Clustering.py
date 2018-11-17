"""
Contain everything related to clustering
"""
import numpy as np
import matplotlib.pyplot as plt
from toolbox_02450 import clusterplot, clusterval
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


class HierarchicalCluster:
    """
    Class to handle everything related to hierarchical clustering in project 3 of 02450 Intro to Machine Learning
    """
    def __init__(self,
                 X=None,
                 y=None,
                 AttributeNames=None,
                 classNames=None,
                 N=None,
                 M=None,
                 C=None,
                 max_cluster=4,
                 method='single',
                 metric='euclidean'):
        """
        init method executed when creating object

        :param X: Feature data
        :param y: True labels
        :param AttributeNames: list of Attrigute names
        :param classNames: list of Class names
        :param N: Number of observations
        :param M: Number of features
        :param C: Number of classes
        :param max_cluster: maximum number of clusters
        :param method: Linkage function, either 'single' 'complete', or 'average', default 'single'
        :param metric: Dissimilarity measure, default 'euclidian'
        """
        print('Clustering object initialized')
        self.X = X
        self.y = y
        self.AttributeNames = AttributeNames
        self.classNames = classNames
        self.N = N
        self.M = M
        self.C = C
        self.method = method
        self.metric = metric
        self.max_cluster = max_cluster
        self.Z = linkage(X, method=self.method, metric=self.metric)

    def _compute_clusters(self, max_cluster):
        cls = fcluster(Z=self.Z, criterion='maxclust', t=max_cluster)
        return cls

    def display_cluster_plot(self, max_cluster=None):
        """
        Display a cluster plot
        :param max_cluster: maximum number of clusters
        :return: plot
        """
        if max_cluster == None:
            max_cluster = self.max_cluster

        cls = self._compute_clusters(max_cluster)
        plt.figure()
        plt.title('Cluster plot with {} clusters and {} linkage'.format(max_cluster, self.method), fontsize=16)
        clusterplot(X=self.X, clusterid=cls.reshape(cls.shape[0], 1), y=self.y)
        plt.show()
        return 'Cluster plot displayed'

    def display_dendogram(self,
                          max_display_levels=10,
                          truncate_method='level',
                          orientation='right',
                          color_threshold=5):
        """
        Display a dendogram
        :param max_display_levels: maximum number of observations in bottom of hierarchy
        :param truncate_method:
        :param orientation: orientation of dendogram, default left-right
        :param color_threshold:
        :return:
        """
        plt.figure()
        plt.title('Dendogram for {} linkage function'.format(self.method), fontsize=16)
        dendrogram(Z=self.Z,
                   truncate_mode=truncate_method,
                   p=max_display_levels,
                   orientation=orientation,
                   color_threshold=color_threshold)
        plt.show()
        return print('Dendrogram displayed')

    def validate_cluster(self):
        #rand, jaccard, NMI = clusterval(self.y, self._compute_clusters(self.max_cluster))
        #out = {'rand':rand, 'jaccard':jaccard, 'NMI':NMI}
        return None #out


if __name__ == '__main__':
    X = np.random.randn(97, 10)
    y = np.random.randn(97, )
    print(X.shape)
    print(y.shape)
    myCluster = HierarchicalCluster(X=X, y=y)
    myCluster.display_cluster_plot(max_cluster=2)
    myCluster.display_dendogram()
    myCluster.validate_cluster()
