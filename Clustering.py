"""
Contain everything related to clustering
"""
import numpy as np
import matplotlib.pyplot as plt
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


class HierarchicalCluster:
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
        print('Clustering object initialized')
        self.X = X
        self.y = y
        self.AttributeNames = AttributeNames
        self.classNames = classNames
        self.N = N
        self.M = M
        self.C = C
        self.max_cluster = max_cluster
        self.Z = linkage(X, method=method, metric=metric)

    def _compute_clusters(self, max_cluster):
        cls = fcluster(Z=self.Z, criterion='maxclust', t=max_cluster)
        return cls

    def display_cluster_plot(self, max_cluster=None):
        if max_cluster == None:
            max_cluster = self.max_cluster

        cls = self._compute_clusters(max_cluster)
        plt.figure(0)
        clusterplot(X=self.X, clusterid=cls.reshape(cls.shape[0], 1), y=self.y)
        plt.show(0)
        return 'Cluster plot displayed'

    def display_dendogram(self,
                          max_display_levels=10,
                          truncate_method='level',
                          orientation='right',
                          color_threshold=5):
        plt.figure(1)
        dendrogram(Z=self.Z,
                   truncate_mode=truncate_method,
                   p=max_display_levels,
                   orientation=orientation,
                   color_threshold=color_threshold)
        plt.show(1)
        return print('Dendrogram displayed')


if __name__ == '__main__':
    X = np.random.randn(97, 10)
    y = np.random.randn(97, )
    print(X.shape)
    print(y.shape)
    myCluster = HierarchicalCluster(X=X, y=y)
    myCluster.display_cluster_plot(max_cluster=2)
    myCluster.display_dendogram()
