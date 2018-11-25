# -*- coding: utf-8 -*-
"""
Contain everything related to outlier detection/Anomaly detection
"""
import numpy as np
from matplotlib.pyplot import (figure, imshow, bar, title, xticks, yticks, cm,
                               subplot, show, plot)
from scipy.io import loadmat
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors


###### Get data using 
import ProstateDataHandler
import seaborn as sns
import report_PCA
import ML_plotter


# Creating object to handle data load and feature transformation
myData = ProstateDataHandler.ProstateData()

raw_data = myData.get_rawData()

# Preparing attribute names
attributeNames = myData.get_attributeNames()
classLabels, classNames, classDict = myData.get_classLabels()
C = len(classNames)

# Generating X  and dimensionality
N, M, X = myData.get_OutlierFeatureData()

X

######

### Gaussian Kernel density

# Estimate the optimal kernel density width, by leave-one-out cross-validation
widths = 2.0**np.arange(-10,10)
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
    print('Fold {:2d}, w={:f}'.format(i,w))
    f, log_f = gausKernelDensity(X, w)
    logP[i] = log_f.sum()
val = logP.max()
ind = logP.argmax()

width=widths[ind]


# Estimate density for each observation not including the observation
# itself in the density estimate
density, log_density = gausKernelDensity(X, width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
density = density[i]

# Plot density estimate of outlier score
figure(1)
bar(range(97),density[:97].reshape(-1,))
title('Density estimate')
figure(2)
plot(logP)
title('Optimal width')
show()
    
print('Optimal estimated width is: {0}'.format(width))

# Display the index of the lowest density data object
print('Gaussian Kernel density')
print('Lowest density: {0} for data object: {1}'.format(density[0],i[0]))
# Display the index of the 2. lowest density data object
print('2. lowest density: {0} for data object: {1}'.format(density[1],i[1]))
# Display the index of the 3. lowest density data object
print('3. lowest density: {0} for data object: {1}'.format(density[2],i[2]))
# Display the index of the 4. lowest density data object
print('4. lowest density: {0} for data object: {1}'.format(density[3],i[3]))
# Display the index of the 5.third lowest density data object
print('5. lowest density: {0} for data object: {1}'.format(density[4],i[4]))

### KNN density

### K-neighbors density estimator
# Neighbor to use:
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

density = 1./(D.sum(axis=1)/K)

# Sort the scores
i = density.argsort()
density = density[i]

# Plot k-neighbor estimate of outlier score (distances)
figure(3)
bar(range(97),density[:97])
title('KNN density: Outlier score')

# Display the index of the lowest density data object
print('K-neighbors density estimator')
print('Lowest density: {0} for data object: {1}'.format(density[0],i[0]))
# Display the index of the 2. lowest density data object
print('2. lowest density: {0} for data object: {1}'.format(density[1],i[1]))
# Display the index of the 3. lowest density data object
print('3. lowest density: {0} for data object: {1}'.format(density[2],i[2]))
# Display the index of the 4. lowest density data object
print('4. lowest density: {0} for data object: {1}'.format(density[3],i[3]))
# Display the index of the 5.third lowest density data object
print('5. lowest density: {0} for data object: {1}'.format(density[4],i[4]))

##### KNN average relative density (ARD)

### K-nearest neigbor average relative density
# Compute the average relative density

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
density = 1./(D.sum(axis=1)/K)
avg_rel_density = density/(density[i[:,1:]].sum(axis=1)/K)

# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]

# Plot k-neighbor estimate of outlier score (distances)
figure(5)
bar(range(97),avg_rel_density[:97])
title('KNN average relative density: Outlier score')

# Display the index of the lowest density data object
print('ARD')
print('Lowest density: {0} for data object: {1}'.format(avg_rel_density[0],i_avg_rel[0]))
# Display the index of the 2. lowest density data object
print('2. lowest density: {0} for data object: {1}'.format(avg_rel_density[1],i_avg_rel[1]))
# Display the index of the 3. lowest density data object
print('3. lowest density: {0} for data object: {1}'.format(avg_rel_density[2],i_avg_rel[2]))
# Display the index of the 4. lowest density data object
print('4. lowest density: {0} for data object: {1}'.format(avg_rel_density[3],i_avg_rel[3]))
# Display the index of the 5.third lowest density data object
print('5. lowest density: {0} for data object: {1}'.format(avg_rel_density[4],i_avg_rel[4]))