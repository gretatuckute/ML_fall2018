"""
Contain everything related to clustering
"""
import numpy as np
import matplotlib.pyplot as plt
from toolbox_02450 import clusterplot, clusterval, gausKernelDensity
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.linalg import svd
from sklearn.mixture import GaussianMixture


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
        :param AttributeNames: list of Attribute names
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

        X = self.X

        Xshape = X.shape

        Xshape = Xshape[1]

        if Xshape > 2:
            X = self._get_principal_components()
            print('X has too many dimensions to plot. PCA performed')
            print(X.shape)



        cls = self._compute_clusters(max_cluster)
        plt.figure()
        plt.title('Cluster plot with {} clusters and {} linkage'.format(max_cluster, self.method), fontsize=16)
        clusterplot(X=X, clusterid=cls.reshape(cls.shape[0], 1), y=self.y)
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

    def validate_cluster(self, K):
        # Allocate variables:
        Rand = np.zeros((K,))
        Jaccard = np.zeros((K,))
        NMI = np.zeros((K,))

        for k in range(K):
            # Compute clusters
            cls = self._compute_clusters(k)
            # compute cluster validities:
            Rand[k], Jaccard[k], NMI[k] = clusterval(self.y, cls)

            # Plot results:

        plt.figure()
        plt.title('Cluster validity using {} linkage'.format(self.method), fontsize=16)
        plt.plot(np.arange(K) + 1, Rand)
        plt.plot(np.arange(K) + 1, Jaccard)
        plt.plot(np.arange(K) + 1, NMI)
        plt.ylim(-2, 1.1)
        plt.legend(['Rand', 'Jaccard', 'NMI'], loc=4)
        plt.show()
        return print('Cluster validity performed')

    def _get_principal_components(self, n_pca=2):
        """
        Performs the singular value decomposition
        :param n_pca: Int, number of PC's returned
        :return: specified number of principal components
        """
        U, S, V = svd(self.X, full_matrices=False)
        Z = np.dot(self.X, V.T)
        return Z[:, 0:n_pca]


class GMM:
    def __init__(self,
                 X=None,
                 y=None,
                 AttributeNames=None,
                 classNames=None,
                 N=None,
                 M=None,
                 C=None,
                 n_components = 4,
                 covariance_type = 'full',
                 n_init = 10):
        """
        init method executed when creating object

        :param X: Feature data
        :param y: True labels
        :param AttributeNames: list of Attribute names
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
        self.K = n_components
        self.cov_type = covariance_type
        self.reps = n_init

    def _fit_GMM(self, ):
        gmm = GaussianMixture(n_components=self.K, covariance_type=self.cov_type, n_init=self.reps).fit(self.X)
        cls = gmm.predict(self.X)
        cds = gmm.means_
        covs = gmm.covariances_

        if self.cov_type.lower() == 'diag':
            new_covs = np.zeros([K, self.M, self.M])

            count = 0
            for elem in covs:
                temp_m = np.zeros([self.M, self.M])
                new_covs[count] = np.diag(elem)
                count += 1

        return cls, cds, covs

    def create_GMM_clusterplot(self, idx=[0,1]):
        cls, cds, covs = self._fit_GMM()

        X = self.X

        Xshape = X.shape

        Xshape = Xshape[1]

        if Xshape > 2:
            X = self._get_principal_components()
            print('X has too many dimensions to plot. PCA performed')
            print(X.shape)

        plt.figure()
        clusterplot(X, clusterid=cls, centroids=cds[:,idx], y=self.y, covars=covs[:,idx,:][:,:,idx])
        plt.show()

    def _get_principal_components(self, n_pca=2):
        """
        Performs the singular value decomposition
        :param n_pca: Int, number of PC's returned
        :return: specified number of principal components
        """
        U, S, V = svd(self.X, full_matrices=False)
        Z = np.dot(self.X, V.T)
        return Z[:, 0:n_pca]





if __name__ == '__main__':
    X = np.random.randn(97, 10)
    y = np.random.randn(97, )
    print(X.shape)
    print(y.shape)
    myCluster = HierarchicalCluster(X=X, y=y)
    myCluster.display_cluster_plot(max_cluster=2)
    myCluster.display_dendogram()
    myCluster.validate_cluster(K=5)


