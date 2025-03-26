import numpy as np

def soft_thresholding(a, b):
    return np.sign(a) * np.maximum(np.abs(a) - b, 0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

class logisitic_regression():
    def __init__(self):
        pass
    
    def set_std_mean(self,X, epsilon = 1e-8):
        self.mean = np.mean(X, axis = 0)
        std = np.std(X,axis = 0)
        self.std = np.where(std > 0, std, epsilon)

    def standarize(self, X):
        return (X-self.mean)/self.std

    def fit(self, X, y, a, epsilon = 0.001, K=100, weights = False, lambdas = None):
        X, y = np.array(X), np.array(y)
        n, p = X.shape
        y = np.array(y)
        self.set_std_mean(X)
        X = self.standarize(X)
        self.B= np.zeros(p)
        q=1/n
        wx2 = 1
        z = q
        if weights:
            z = 0.25
        if user_lambda :
            lambdas = np.repeat(user_lambda,K)
        else: 
            lambda_max= np.max(np.abs((y- 0.5)@X*z)) #since B = 0 p is 0.5 and w is 0.25 everywhere
            if a != 0:
                lambda_max /= a
            lambdas = np.logspace(np.log10(lambda_max), np.log10(0.001*lambda_max), K)
            lambdas = np.logspace(np.log10(lambda_max), np.log10(epsilon*lambda_max), K)
        count = 0
        print(len(lambdas) * g * p)
        for lambd in lambdas:
            for k in range(g):
                for j in range(p):
                    if count % 1000 == 0:
                        print(f'Count {count}')                    
                    w_sum = 1
                    w_sumx2 = 1
                    xj = X[:,j]
                    preds = self.predict_proba(X)[:,k]
                    w =  preds*(1-preds)
                    if weights:
                        w_sumx2 = w@(xj*xj)
                        xj = w*xj

                    # old version just in case    
                    #sum = (xj@(y[:,k])) - xj@X@(self.B[:,k]) + w_sum*self.B[j,k]
                    #sum = xj@(y[:,k]-self.predict_proba(X)[:,k])
                    #Implemented with using z as y
                    sum = -xj@(xj*w*self.B[j,k]- y[:,k] +preds)
                    self.B[j,k]= soft_thresholding(sum/n,lambd*a)/(w_sumx2+lambd*(1-a))
                    
                    count += 1
                    if count > max_count:
                        break

    def predict_proba(self, X):
        X = np.array(X)
        X = self.standarize(X)
        X = np.exp(X@self.B)
        return X / X.sum(axis=1, keepdims=True)
    
            for j in range(p):
                preds = sigmoid(X@self.B) 
                w = preds*(1-preds)
                xj = (X[:,j]).reshape((n,1))
                if weights:
                    #p and wx2 has different forms depends on version that we choose
                    q=w
                    wx2 = (w @ (xj**2))[0]
                sum = (q*w*X[:,j]*self.B[j] +q*(y-preds))@xj
                self.B[j] = soft_thresholding(sum[0],lambd*a)/(wx2 +lambd*(1-a))
        
    def predict(self, X):
        X = self.standarize(X)
        return np.round(1/(1+np.exp(-X@self.B)))           

        