'''
README
 
This sit he main file which calls all other funcitons like greedy and ten_fold.
If you run this file, you see the output for both algorithms

'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from greedy import *
from ridge import *

####Q2
# read data file
location=('./train-matrix.txt')

#read data file train-matrix.txt into a dataframe
header_details=pd.read_table(location,nrows=2,header=None)
no_samples=header_details[0][0]
no_features=header_details[0][1]

X_DF=pd.read_table(location,sep=' ',header=None,skiprows=2,nrows=no_samples)
Y_DF=pd.read_table(location,sep=' ',header=None,skiprows=1002)
#print(X_DF.shape,Y_DF.shape)



X=np.array(X_DF.values)
Y=np.array(Y_DF.values)
X_ridge=np.insert(X,0,1,axis=1)

#Greedy Algorithm
A,beta=greedy(X,Y,6)
print('Greedy beta:',beta)
print('\n\n')

#Ridge Regression
Lambdas=[0.0125,0.025,0.05,0.1,0.2]
beta_ridge,lamda_opt=ten_fold (X_ridge, Y, Lambdas)
print("Ridge estimator on train set, Beta and best Lambda: ",beta_ridge,lamda_opt)


####Q3
#read test matrix
# read data file
location=('./test-matrix.txt')

#read data file train-matrix.txt into a dataframe
header_details=pd.read_table(location,nrows=2,header=None)
no_samples2=header_details[0][0]
no_features2=header_details[0][1]
X_DF_test=pd.read_table(location,sep=' ',header=None,skiprows=2,nrows=no_samples2)
Y_DF_test=pd.read_table(location,sep=' ',header=None,skiprows=12)


Xtest=np.array(X_DF.values)
Ytest=np.array(Y_DF.values)

Xtest=np.insert(X,0,1,axis=1)
test_beta,l=ten_fold(Xtest,Ytest,[0.0125])
print('Beta estimated for test matrix: ', test_beta)

###Q3
#Compute prediction error compared with true beta
location=('./true-beta.txt')
header_details=pd.read_table(location,nrows=1,header=None)
true_beta=pd.read_table(location,header=None,skiprows=1)
true_beta=np.array(true_beta.values)
pred_Error=np.linalg.norm(test_beta[1:]-true_beta,2)**2

print(pred_Error)