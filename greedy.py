import numpy as np
def greedy (X, y, K):
	no_features=np.shape(X)[1]
	beta=np.zeros((no_features,1))
	A=[]
	for k in range(K):
	    res=0
	    max_i=-1
	    for j in range(no_features):
	        temp=abs(np.dot(X[:,j],np.dot(X,beta)-y))
	        if (temp>res):
	            res=temp
	            max_i=j
	    A.append(max_i)

	    xx=[]

	    for index in range(len(A)):
	    	xx.append(np.copy(X[:,A[index]]))

	    xxx=np.array(xx)
	    xxx=xxx.T

	    b=np.dot(np.dot(np.linalg.inv(np.dot(xxx.T,xxx)),xxx.T),y)

	    i=0

	    for col in A:
		    beta[col]=b[i]
		    i+=1
	return A, beta
