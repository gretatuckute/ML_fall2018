import seaborn as sns
import os
import pandas as pd
from scipy.stats import zscore
from categoric2numeric import *
from similarity import binarize2
import numpy as np
import matplotlib.pyplot as plt
from toolbox_02450 import clusterplot, clusterval, gausKernelDensity
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from scipy.linalg import svd
from apyori import apriori
from matplotlib.pyplot import (figure, imshow, bar, title, xticks, yticks, cm,
                               subplot, show, plot)
from sklearn.neighbors import NearestNeighbors

# Nicer formatting of plots
sns.set_style("darkgrid")

# Controllers
set_feature_plot = False
set_hierarchical_clustering = False
set_GMM_clustering = True
set_display_pca = False
set_association_mining = False
set_outlier_detection = False



class ProstateData:
    """
    Class to handle the Prostate data
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
        attributeNames = ['lCaVol', 'lWeight', 'Age', 'lBPH', 'lCP', 'lPSA', 'Gleason 6.0', 'Gleason 7.0',
                          'Gleason 8.0', 'Gleason 9.0']
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
        X_classification = X_k  # [:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        y_classification = svi.squeeze()  # X_k[:, 10]
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
        # X = np.delete(X, 4, 1)  # Deletes SVI
        X_z = zscore(X)
        [X_Gleason, attribute_names_Gleason] = categoric2numeric(gleason)
        svi = np.reshape(svi, [97, 1])
        X_k = np.concatenate((X_z, X_Gleason), axis=1)
        N, M = X_k.shape
        print('There are {} features in the data set'.format(M))
        print('There are {} observations in the data set'.format(N))
        print('X has shape {}'.format(np.shape(X_k)))

        return N, M, X_k

    def get_binarizedFeatureData(self):
        _, _, X, y, _ = self.get_ClassificationFeatureData()
        attributeNames = self.get_attributeNames()
        gleason = X[:, 6:]
        X = X[:, 0:6]

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
    """
    Class to handle everything related to Gaussian Mixture Model
    """
    def __init__(self,
                 X=None,
                 y=None,
                 AttributeNames=None,
                 classNames=None,
                 N=None,
                 M=None,
                 C=None,
                 n_components = 7,
                 covariance_type = 'full',
                 n_init = 10,
                 tolerance = 1e-6):
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
        self.tol = tolerance

    def _fit_GMM(self, ):
        gmm = GaussianMixture(n_components=self.K, covariance_type=self.cov_type, n_init=self.reps, tol=self.tol,reg_covar=0.01).fit(self.X)
        cls = gmm.predict(self.X)
        cds = gmm.means_
        covs = gmm.covariances_

        if self.cov_type.lower() == 'diag':
            new_covs = np.zeros([self.K, self.M, self.M])

            count = 0
            for elem in covs:
                temp_m = np.zeros([self.M, self.M])
                new_covs[count] = np.diag(elem)
                count += 1

                covs = new_covs

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

    def cross_validation(self, k_range=range(1,11), n_splits=5):
        T = len(k_range)

        # Allocate variables
        BIC = np.zeros((T,))
        AIC = np.zeros((T,))
        CVE = np.zeros((T,))
        
        COV = np.zeros((T,))

        # K-fold cross validation
        CV = KFold(n_splits=n_splits, shuffle=True)

        for t,K in enumerate(k_range):
            print('Fitting model for K={0}'.format(K))

            gmm = GaussianMixture(n_components=K, covariance_type=self.cov_type, n_init=self.reps).fit(self.X)

            BIC[t,] = gmm.bic(self.X)
            AIC[t,] = gmm.aic(self.X)


            for train_index, test_index in CV.split(self.X):

                X_train = self.X[train_index]
                X_test = self.X[test_index]

                gmm = GaussianMixture(n_components=K, covariance_type=self.cov_type, n_init=self.reps,reg_covar=0.01).fit(X_train)

                CVE[t] += -gmm.score_samples(X_test).sum()
                

        plt.figure()
        plt.plot(k_range, BIC, '-*b')
        plt.plot(k_range, AIC, '-xr')
        plt.plot(k_range, 2 * CVE, '-ok')
        plt.legend(['BIC', 'AIC', 'CV (neg. log likelihood)'])
        plt.xlabel('K')
        plt.show()


class prostatePCA:
    """
    Class to perform PCA on prostate data
    """

    def __init__(self, X=None):
        print('Performing PCA on the Prostate Dataset')
        self.X = X

    def get_principal_components(self, n_pca=2):
        """
        Performs the singular value decomposition
        :param n_pca: Int, number of PC's returned
        :return: specified number of principal components
        """
        U, S, V = svd(self.X, full_matrices=False)
        Z = np.dot(self.X, V.T)
        return Z[:, 0:n_pca]

    def display_pca(self):
        """
        Display the first two prinsipal components against each other
        :return: plot
        """
        Z = self.get_principal_components()
        plt.figure()
        plt.scatter(Z[:, 0], Z[:, 1])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PC1 vs PC2')
        plt.show()

    def display_pca_outlier_detection(self):
        """
        Display the first two prinsipal components against each other with outliers
        :return: plot
        """
        Z = self.get_principal_components()
        plt.figure()
        plt.scatter(Z[:, 0], Z[:, 1])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PC1 vs PC2')
        # KDE
        # outliers = np.array([2, 46, 88, 56, 48])
        # KNN
        # outliers = np.array([93, 2, 88, 56, 46])
        # ARD
        outliers = np.array([2, 46, 11, 68, 88])
        for i in outliers:
            plt.annotate(i, (Z[i, 0], Z[i, 1]))
        plt.show()


class Outlier_Detection:
    def __init__(self,
                 X=None,
                 AttributeNames=None,
                 classNames=None,
                 N=None,
                 M=None,
                 C=None,
                 K=5):
        """
        init method executed when creating object

        :param X: Feature
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
        print('Outlier detection object initialized')
        self.X = X
        self.AttributeNames = AttributeNames
        self.classNames = classNames
        self.N = N
        self.M = M
        self.C = C
        self.K = K

    def gaussian_kernel_density(self):
        # Estimate the optimal kernel density width, by leave-one-out cross-validation
        widths = 2.0 ** np.arange(-10, 10)
        logP = np.zeros(np.size(widths))
        for i, w in enumerate(widths):
            print('Fold {:2d}, w={:f}'.format(i, w))
            f, log_f = gausKernelDensity(self.X, w)
            logP[i] = log_f.sum()
        val = logP.max()
        ind = logP.argmax()

        width = widths[ind]

        # Estimate density for each observation not including the observation
        # itself in the density estimate
        density, log_density = gausKernelDensity(self.X, width)

        # Sort the densities
        i_KDE = (density.argsort(axis=0)).ravel()
        density = density[i_KDE]

        # Plot density estimate of outlier score
        figure()
        bar(range(97), density[:97].reshape(-1, ))
        title('Gaussian Kernel Density: Outlier score')
        show()

        figure()
        plot(logP)
        title('Optimal width')
        show()

        print('Optimal estimated width is: {0}'.format(width))

        # Display the index of the lowest density data object
        print('Gaussian Kernel density')
        print('Lowest density: {0} for data object: {1}'.format(density[0], i_KDE[0]))
        # Display the index of the 2. lowest density data object
        print('2. lowest density: {0} for data object: {1}'.format(density[1], i_KDE[1]))
        # Display the index of the 3. lowest density data object
        print('3. lowest density: {0} for data object: {1}'.format(density[2], i_KDE[2]))
        # Display the index of the 4. lowest density data object
        print('4. lowest density: {0} for data object: {1}'.format(density[3], i_KDE[3]))
        # Display the index of the 5. lowest density data object
        print('5. lowest density: {0} for data object: {1}'.format(density[4], i_KDE[4]))
        print(density)

        return i_KDE

    def knn_density(self):


        # Find the k nearest neighbors
        knn = NearestNeighbors(n_neighbors=self.K).fit(self.X)
        D, i = knn.kneighbors(self.X)

        density = 1. / (D.sum(axis=1) / self.K)

        # Sort the scores
        i_KNN = density.argsort()
        density = density[i_KNN]

        # Plot k-neighbor estimate of outlier score (distances)
        figure()
        bar(range(97), density[:97])
        title('KNN density: Outlier score')
        show()

        # Display the index of the lowest density data object
        print('K-neighbors density estimator')
        print('Lowest density: {0} for data object: {1}'.format(density[0], i_KNN[0]))
        # Display the index of the 2. lowest density data object
        print('2. lowest density: {0} for data object: {1}'.format(density[1], i_KNN[1]))
        # Display the index of the 3. lowest density data object
        print('3. lowest density: {0} for data object: {1}'.format(density[2], i_KNN[2]))
        # Display the index of the 4. lowest density data object
        print('4. lowest density: {0} for data object: {1}'.format(density[3], i_KNN[3]))
        # Display the index of the 5.third lowest density data object
        print('5. lowest density: {0} for data object: {1}'.format(density[4], i_KNN[4]))
        print(density)

        return i_KNN

    def knn_ard(self):
        knn = NearestNeighbors(n_neighbors=self.K).fit(self.X)
        D, i = knn.kneighbors(self.X)
        density = 1. / (D.sum(axis=1) / self.K)
        avg_rel_density = density / (density[i[:, 1:]].sum(axis=1) / self.K)

        # Sort the avg.rel.densities
        i_avg_rel = avg_rel_density.argsort()
        avg_rel_density = avg_rel_density[i_avg_rel]

        i_KDE = self.gaussian_kernel_density()
        i_KNN = self.knn_density()

        # Plot k-neighbor estimate of outlier score (distances)
        figure()
        bar(range(97), avg_rel_density[:97])
        title('KNN average relative density: Outlier score')
        show()

        # Display the index of the lowest density data object
        print('ARD')
        print('Lowest density: {0} for data object: {1}'.format(avg_rel_density[0], i_avg_rel[0]))
        # Display the index of the 2. lowest density data object
        print('2. lowest density: {0} for data object: {1}'.format(avg_rel_density[1], i_avg_rel[1]))
        # Display the index of the 3. lowest density data object
        print('3. lowest density: {0} for data object: {1}'.format(avg_rel_density[2], i_avg_rel[2]))
        # Display the index of the 4. lowest density data object
        print('4. lowest density: {0} for data object: {1}'.format(avg_rel_density[3], i_avg_rel[3]))
        # Display the index of the 5.third lowest density data object
        print('5. lowest density: {0} for data object: {1}'.format(avg_rel_density[4], i_avg_rel[4]))
        print(density)
        #
        print('KDE')
        print(i_KDE)
        print('KNN')
        print(i_KNN)
        print('ARD')
        print(i_avg_rel)

    def pca_outlier(self):
        prostatePC = prostatePCA(X=self.X)

        # Projecting the data down to the first two principal components (easier plotting)
        X_pca = prostatePC.get_principal_components(n_pca=2)

        # Displaying the first two principal components against each other
        prostatePC.display_pca_outlier_detection()

        return print('PCA outlier plot')


def plot_attributes_2d(X, y, C, classNames, attributeNames, i=0, j=1, saveFigure=False):
    """
    Method for plotting different attributes against each other
    :param X:
    :param y:
    :param C:
    :param classNames:
    :param attributeNames:
    :param i:
    :param j:
    :return:
    """
    # Plotting the data set (different attributes to be specified)
    plt.figure()

    for c in range(C):
        # select indices belonging to class c:
        class_mask = y==c
        plt.plot(X[class_mask,i], X[class_mask,j], 'o')

    #plt.legend(classNames)
    svi_legend = ['SVI 0', 'SVI 1']
    plt.legend(svi_legend)#, loc='upper center', ncol=1)#, bbox_to_anchor=(0.5, -0.3))
    plt.rc('legend',fontsize=24)
    axis_font = {'size':'24'}
    plt.xlabel(attributeNames[i], **axis_font)
    plt.ylabel(attributeNames[j], **axis_font)
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)

    if saveFigure == True:
        plt.savefig('./ML_fall2018/Figures/' + attributeNames[i] +  "_vs_" + attributeNames[j]+".png")

    # Output result to screen
    plt.show()


class AssociationMining:
    def __init__(self):
        print('Association mining object created')

    def mat2transactions(self, X, labels=[]):
        T = []
        for i in range(X.shape[0]):
            l = np.nonzero(X[i, :])[0].tolist()
            if labels:
                l = [labels[i] for i in l]
            T.append(l)
        return T

    def get_rules(self, t, min_support=0.8, min_confidence=1, print_rules=False):
        rules = apriori(t, min_support=min_support, min_confidence=min_confidence)

        if print_rules:
            frules = []
            for r in rules:
                conf = r.ordered_statistics[0].confidence
                supp = r.support
                x = ', '.join(list(r.ordered_statistics[0].items_base))
                y = ', '.join(list(r.ordered_statistics[0].items_add))
                print('{%s} -> {%s}  (supp: %.3f, conf: %.3f)' % (x, y, supp, conf))
        return rules






# Creating object to handle data load and feature transformation
myData = ProstateData()

raw_data = myData.get_rawData()


# Preparing attribute names
attributeNames = myData.get_attributeNames()
classLabels, classNames, classDict = myData.get_classLabels()
C = len(classNames)




# Plotting two features against each other
if set_feature_plot:
    # Generating X & Y and dimensionality
    N, M, X, y, svi = myData.get_ClassificationFeatureData()
    plot_attributes_2d(i=2,
                        j=5,
                        X=X,
                        y=y,
                        C=C,
                        classNames=classNames,
                        attributeNames=attributeNames,
                        saveFigure=False)


if set_hierarchical_clustering:

    # Generating X & Y and dimensionality
    N, M, X, y, svi = myData.get_ClassificationFeatureData()

    # Creating a PCA object
    prostatePC = prostatePCA(X=X)

    # Projecting the data down to the first two principal components (easier plotting)
    X_pca = prostatePC.get_principal_components(n_pca=2)

    # Displaying the first two principal components against each other
    if set_display_pca:
        prostatePC.display_pca()

    # Perform hierarchical clustering
    prostate_hierarchical_clustering = HierarchicalCluster(X=X_pca, y=y, method='single')
    prostate_hierarchical_clustering.display_cluster_plot(max_cluster=2)
    prostate_hierarchical_clustering.display_dendogram(max_display_levels=100, orientation='top', color_threshold=7.0)
    prostate_hierarchical_clustering.validate_cluster(K=10)


if set_GMM_clustering:
    # Generating X & Y and dimensionality
    N, M, X, y, svi = myData.get_ClassificationFeatureData()

    # Creating a PCA object
    prostatePC = prostatePCA(X=X)

    # Projecting the data down to the first two principal components (easier plotting)
    X_pca = prostatePC.get_principal_components(n_pca=2)

    prostateGMM = GMM(X=X, y=y, n_components=2)
    prostateGMM.create_GMM_clusterplot()
    prostateGMM.cross_validation()


if set_association_mining:
    Xbin, attributeNamesBin = myData.get_binarizedFeatureData()
    prostate_association_mining = AssociationMining()
    transactions = prostate_association_mining.mat2transactions(Xbin, attributeNamesBin)
    rules = prostate_association_mining.get_rules(t=transactions, min_support=0.17, min_confidence=0.8, print_rules=True)


if set_outlier_detection:
    N, M, X = myData.get_OutlierFeatureData()
    prostateOutlier = Outlier_Detection(X=X)
    prostateOutlier.gaussian_kernel_density()
    prostateOutlier.knn_density()
    prostateOutlier.knn_ard()
    prostateOutlier.pca_outlier()
