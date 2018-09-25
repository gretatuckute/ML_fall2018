import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from similarity import similarity

"""
The purpose of this application is to inspect the prostate dataset that we intend
to use for the project in 02450 Intro to Machine Learning

To-do:
We need to fix the indexing of the columns - the way it is implemented right now something strange happens
Encode certain columns differently, e.g. lcp and lbsa

Author: Peter Bakke
Reviewed by: Greta Tuckute hej
Last modified: 16/09/18, 17.40
"""


def DataLoader(path, sheet):
    """
    Method for importing data from a spreadsheet.

    :param path: full path to the spreadsheet to load
    :param sheet: name of the sheet in the workbook that is loaded
    :return: pandas dataFrame with imported data
    """
    import pandas as pd

    out = pd.read_excel(path, sheet_name=sheet)

    return out


# Specify path and sheet name in the prostate workbook
filePath = 'C:/Users/PeterBakke/Documents/git/ML_fall2018/Data/Prostate.xlsx'
#filePath = 'C:/Users/Greta/Documents/Github/ML_fall2018/Data/Prostate.xlsx'
#filePath = 'C:/Users/narisa/Documents/GitHub/ML_fall2018/Data/Prostate.xlsx'
sheet = 'Sheet1'

# load prostate data into dataFrame
myData = DataLoader(path=filePath, sheet=sheet)

# delete irrelevant columns
del myData['ID']
del myData['train']

# extract class names and encode with integers (dict)
attributeNames = list(myData.columns.values)
classLabels = myData['gleason'].values.tolist()
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(4)))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Convert dataFrame to numpy array
X = myData.values

# Compute values of N, M and C
N = len(y)
M = len(attributeNames)
C = len(classNames)

# Data attributes to be plotted
i = 0
j = 8

# Plotting the data set (different attributes to be specified)
f = plt.figure()
plt.title('Prostate data of attributes: ' + str(attributeNames[i]) + ' vs. ' + str(attributeNames[j]))

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(X[class_mask,i], X[class_mask,j], 'o')

#plt.legend(classNames)
gleason_legend = ['Gleason Score 6', 'Gleason Score 7', 'Gleason Score 8', 'Gleason Score 9']
plt.legend(gleason_legend)
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])

# Output result to screen
plt.show()

#Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

Z = np.dot(Y, V.T)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

print(rho)

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'o-')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.show()

# Indices of the principal components to be plotted
ii = 0
jj = 1

# Plot PCA of the data
f = plt.figure()
plt.title('Prostate data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(Z[class_mask,ii], Z[class_mask,jj], 'o')
plt.legend(classNames)
plt.xlabel('PC{0}'.format(ii+1))
plt.ylabel('PC{0}'.format(jj+1))


# Plot PCA_ii of the data against lpsa
f = plt.figure()
plt.title('Prostate data: PCA against lpsa')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(Z[class_mask,ii], Y[class_mask,8], 'o')
plt.legend(classNames)
plt.xlabel('PC{0}'.format(ii+1))
plt.ylabel('lpsa')


# Output result to screen
plt.show()





# Make Boxplots
plt.figure()
plt.boxplot(Y)
plt.title('Boxplots of demeaned data');
plt.xticks([1,2,3,4,5,6,7,8],attributeNames)
plt.grid(axis='y',linestyle='dashed')
plt.show()

# Summary statistics
# Create dict of attribute dictonaries using comprehensions
keys = ['mean', 'std', 'median', 'range', 'Q_25', 'Q_75']

statistics={name:{key:[] for key in keys} for name in attributeNames}


for attribute in statistics:
    k = attributeNames.index(attribute)
    statistics[attribute]['mean'] = X[:,k].mean()
    statistics[attribute]['std'] = X[:,k].std(ddof=1)
    statistics[attribute]['median'] = np.median(X[:,k])
    statistics[attribute]['range'] = X[:,k].max()-X[:,k].min()
    statistics[attribute]['Q_25'] = np.percentile(X[:,k],25)
    statistics[attribute]['Q_75'] = np.percentile(X[:,k],75)

# Create Boxplot of raw data
plt.figure()
plt.boxplot(X)
plt.title('Boxplots of raw data');
plt.xticks([1,2,3,4,5,6,7,8],attributeNames)
plt.grid(axis='y',linestyle='dashed')
plt.show()    

#Covariance and correlation
# Covariance of X
covariance_X = np.cov(X)
#print(covariance_X)
#correlation of X
correlation_X = numpy.corrcoef(X)
#print(correlation_X)

#Similatiry

# Attribute to use as query

# Similarity: 'SMC', 'Jaccard', 'ExtendedJaccard', 'Cosine', 'Correlation' 
similarity_measure = 'cos'

N, M = X.shape
# Search for similar attributes
# Index of all other attributes than i

for i in [0,1,2,3,4,5,6,7,8]:
    noti = list(range(0,i)) + list(range(i+1,M)) 
    # Compute similarity between attribute i and all others
    sim = similarity(X[:,i], X[:,noti].T, similarity_measure)
    sim = sim.tolist()[0]
    # Tuples of sorted similarities and their attribute name
    Name = []
    for number in noti:
        Name.append(attributeNames[number])
    sim_to_index = sorted(zip(sim, Name))
    print('Similarity of ', attributeNames[i], 'to:')
    print(sim_to_index)

# Calculate projections of Y on Eqigenvector
plt.figure()
plt.boxplot(Y)
plt.title('Boxplots of demeaned data');
plt.xticks([1,2,3,4,5,6,7,8],attributeNames)
plt.grid(axis='y',linestyle='dashed')
plt.show()

print(V[:,1].T)
## Projection of water class onto the 2nd principal component.

# When Y and V have type numpy.array, then @ is matrix multiplication
print( Y[y==4,:] @ V[:,1] )

# or convert V to a numpy.mat and use * (matrix multiplication for numpy.mat)
#print((Y[y==4,:] * np.mat(V[:,1]).T).T)



print('Ran Exercise 2.1.5')