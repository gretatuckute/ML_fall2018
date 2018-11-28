# exercise 2.2.4
from ex2_1_1 import *
from scipy.linalg import svd

# (requires data structures from ex. 2.2.1 and 2.2.3)
Y = X - np.ones((N,1))*X.mean(0)
U,S,V = svd(Y,full_matrices=False)
V=V.T


print(V[:,1].T)
## Projection of water class onto the 2nd principal component.

# When Y and V have type numpy.array, then @ is matrix multiplication
print( Y[y==4,:] @ V[:,1] )

# or convert V to a numpy.mat and use * (matrix multiplication for numpy.mat)
#print((Y[y==4,:] * np.mat(V[:,1]).T).T)


print('Ran Exercise 2.1.5')