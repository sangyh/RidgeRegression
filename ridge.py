import numpy as np

def ridge (X, y, Lambda):
	I = np.identity(X.shape[1])
	n=X.shape[0]
	beta=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)/(2*n)+Lambda*I),X.T/(2*n)),y)
	return beta

def ten_fold (X, y, Lambdas):
	Kfold=10
	interval=np.shape(X)[0]/Kfold
	segments=[int(interval*x) for x in range(Kfold)]

	min_error=10000000000000000

	for l in Lambdas:
		mean_error=[]
		beta_array=[]
		for fold in range(Kfold):   
			train_X=np.delete(X,slice(segments[fold]+1,segments[fold]+100,None),axis=0)
			test_X=X[segments[fold]+1:segments[fold]+100]
			train_y=np.delete(y,slice(segments[fold]+1,segments[fold]+100,None),axis=0)
			test_y=y[segments[fold]+1:segments[fold]+100]
			beta_array.append(ridge(train_X,train_y,l))
			
			y_pred=np.dot(test_X,beta_array[fold]) #+l*np.linalg.norm(b,2)**2
			pred_error=0
			pred_error=np.sum(((np.subtract(y_pred,test_y))**2))/len(y_pred)

			mean_error.append(pred_error) #collect mean errros for each fold

		
		print('mean error and lamb:',np.mean(mean_error),l)
		if np.mean(mean_error)<min_error:
			min_error=np.mean(mean_error)
			Lambda=l
			beta=np.mean(beta_array,axis=0)

	
	return beta, Lambda
