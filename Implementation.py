import numpy as np
from sklearn.model_selection import train_test_split
import measures

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
    
    def fit(self, X, y, max_iter=100, a = 1,weights = False, user_lambda = None, fit_intercept = True):
        X= np.array(X)
        n, p = X.shape
        y = np.array(y)
        self.set_std_mean(X)
        X = self.standarize(X)
        self.B= np.zeros(p)
        q=1/n
        wx2 = 1
        z = q
        if fit_intercept:
            prior = y.mean()
        else:
            prior= 0.5
        self.B0 = np.log(prior/(1-prior))
        if weights:
            z = prior*(1-prior)
        if not user_lambda: 
            lambda_max= np.max(np.abs((y- prior)@X*z)) #since B = 0 w is 0.5 everywhere so this is the biggest possible value
            if a != 0:
                lambda_max /= a
            lambdas = np.logspace(np.log10(lambda_max), np.log10(0.001*lambda_max), max_iter)
        else:
            lambdas = np.repeat(user_lambda,max_iter)
            
        for lambd in lambdas:
            for j in range(p):
                preds = sigmoid(X@self.B+ self.B0) 
                w = preds*(1-preds)
                xj = (X[:,j]).reshape((n,1))
                if weights:
                    #p and wx2 has different forms depends on version that we choose
                    q=w
                    wx2 = (w @ (xj**2))[0]
                sum = (q*w*X[:,j]*self.B[j] +q*(y-preds))@xj
                self.B[j] = soft_thresholding(sum[0],lambd*a)/(wx2 +lambd*(1-a))
        
    def predict_proba(self,X):
        X = self.standarize(X)
        return sigmoid(X@self.B+ self.B0) 
    
    def predict(self, X):
        X = self.standarize(X)
        return np.round(sigmoid(X@self.B + self.B0))        
    
    def validate(self, X_valid, y_valid, measure):
        return measure(self.fit(X_valid), y_valid)

    def plot(self, measure, X, y, lambdas = None, max_iter = 200):
        metrics = {
            "precision": measures.precision,
            "recall":  measures.recall,
            "f_measure":  measures.f_measure,
            "balanced_accuracy":  measures.balanced_accuracy,
            "auc_roc":  measures.auc_roc,
            "auc_pr":  measures.auc_pr
        }
        metric = metrics.get(measure)
        X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        #just for testing if works can delete it
        if not lambdas:
            n= X.shape[0]
            z = 1/n
            prior = y.mean()
            lambda_max= np.max(np.abs((y- prior)@X*z)) 
            lambdas = np.linspace(lambda_max-0.001, 0.001*lambda_max, num=100)
        results = []
        for lambd in lambdas:
            self.fit(X, y, max_iter=max_iter, user_lambda=lambd)
            results.append(self.validate(X_val, y_val, metric))
        
        # Plot to be done

    def plot_coefficients(self, X, y, lambdas = None, max_iter = 200):
        if not lambdas:
            n= X.shape[0]
            z = 1/n
            prior = y.mean()
            lambda_max= np.max(np.abs((y- prior)@X*z)) 
            lambdas = np.linspace(lambda_max-0.001, 0.001*lambda_max, num=100)
        results = []
        for lambd in lambdas:
            self.fit(X, y, max_iter=max_iter, user_lambda=lambd)
            results.append(self.B)

        # Plot to be done