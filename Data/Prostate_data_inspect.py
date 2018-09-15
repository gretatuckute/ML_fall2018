import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd


"""
The purpose of this application is to inspect the prostate dataset that we intend
to use for the project in 02450 Intro to Machine Learning

To-do:
We need to fix the indexing of the columns - the way it is implemented right now something strange happens


Author: Peter Bakke
Reviewed by: xxx
Last modified: 
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
filePath = 'C:/Users/PeterBakke/Documents/dtu/02450-intro-to-machine-learning/prostate.xlsx'
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
i = 3
j = 8



# Make another more fancy plot that includes legend, class labels,
# attribute names, and a title.
f = plt.figure()
plt.title('Prostate data')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(X[class_mask,i], X[class_mask,j], 'o')

plt.legend(classNames)
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

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'o-')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.show()


# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
plt.title('Prostate data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o')
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()

