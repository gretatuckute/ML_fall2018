# exercise 8.3.2 Fit Multinomial logistic regression
from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from toolbox_02450 import dbplotf
import numpy as np
import sklearn.neural_network as nn
import sklearn.linear_model as lm

# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/synth3.mat')
X = mat_data['X']
X = X - np.ones((X.shape[0],1)) * np.mean(X,0)
X_train = mat_data['X_train']
X_test = mat_data['X_test']
y = mat_data['y'].squeeze()
y_train = mat_data['y_train'].squeeze()
y_test = mat_data['y_test'].squeeze()

#attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)
NHiddenUnits = 2; # <-- Try to change this, what happens? why?

#%% Model fitting and prediction
## ANN Classifier, i.e. MLP with one hidden layer
clf = nn.MLPClassifier(solver='lbfgs',alpha=1e-4,
                       hidden_layer_sizes=(NHiddenUnits,), random_state=1)
clf.fit(X_train,y_train)
print('Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(np.sum(clf.predict(X_test)!=y_test),len(y_test)))


# Multinomial logistic regression
logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, random_state=1)
logreg.fit(X_train,y_train)
# Number of miss-classifications
print('Number of miss-classifications for Multinormal regression:\n\t {0} out of {1}'.format(np.sum(logreg.predict(X_test)!=y_test),len(y_test)))

#%% Decision boundaries for the ANN model
figure(1)
def neval(xval):
    return np.argmax(clf.predict_proba(xval),1)

dbplotf(X_test,y_test,neval,'auto')
show()

#%% Decision boundaries for the multinomial regression model
figure(1)
def nevallog(xval):
    return np.argmax(logreg.predict_proba(xval),1)

dbplotf(X_test,y_test,nevallog,'auto')
show()

print('Ran Exercise 8.3.2')