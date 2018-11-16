"""
Create a PCA decomposition of Prostate data
"""

from scipy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt


class prostatePCA:
    def __init__(self, X=None):
        print('Performing PCA on the Prostate Dataset')
        self.X = X

    def get_principal_components(self, n_pca=2):
        U, S, V = svd(self.X, full_matrices=False)
        Z = np.dot(self.X, V.T)
        return Z[:,0:n_pca]

    def display_pca(self):
        Z = self.get_principal_components()
        plt.figure()
        plt.scatter(Z[:,0],Z[:,1])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PC1 vs PC2')
        plt.show()



if __name__ == '__main__':
    X = np.random.randn(97, 10)
    myPCA = prostatePCA(X=X)
    X_projected = myPCA.get_principal_components(n_pca=2)
    print(X_projected.shape)
    myPCA.display_pca()