import numpy as np

def soft_thresholding(a, b):
    return np.sign(a) * np.maximum(np.abs(a) - b, 0)

def one_hot_encode(y, num_classes=None):
    y = np.array(y)
    if num_classes is None:
        num_classes = np.max(y) + 1
    
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


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
        self.set_std_mean(X)
        X = self.standarize(X)
        y = one_hot_encode(y)
        g = y.shape[1]
        self.B = np.zeros((p, g))
        if not lambdas: 
            lambda_max= np.max(np.abs(X.T@y/n))
            if a != 0:
                lambda_max /= a
            lambdas = np.logspace(np.log10(lambda_max), np.log10(epsilon*lambda_max), K)

        for lambd in lambdas:
            for k in range(g):
                for j in range(p):
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

    def predict_proba(self, X):
        X = np.array(X)
        X = self.standarize(X)
        X = np.exp(X@self.B)
        return X / X.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        predictions = self.predict_proba(X)
        return np.argmax(predictions, axis = 1)
        