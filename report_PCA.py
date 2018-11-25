"""
Create a PCA decomposition of Prostate data
"""

from scipy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt


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
        return Z[:,0:n_pca]

    def display_pca(self):
        """
        Display the first two prinsipal components against each other
        :return: plot
        """
        Z = self.get_principal_components()
        plt.figure()
        plt.scatter(Z[:,0],Z[:,1])
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
        plt.scatter(Z[:,0],Z[:,1])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PC1 vs PC2')
        #KDE
        outliers = np.array([2, 46, 88, 56, 48])          
        #KNN
        #outliers = np.array([93, 2, 88, 56, 46])
        #ARD
        #outliers = np.array([2, 46, 11, 68, 88])
        for i in outliers:
            plt.annotate(i, (Z[i,0],Z[i,1]))
        plt.show()


if __name__ == '__main__':
    X = np.random.randn(97, 10)
    myPCA = prostatePCA(X=X)
    X_projected = myPCA.get_principal_components(n_pca=2)
    print(X_projected.shape)
    myPCA.display_pca()