import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

def soft_thresholding(a, b):
    """Soft thresholding used in fitting of model"""
    return np.sign(a) * np.maximum(np.abs(a) - b, 0)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))

class logisitic_regression():
    """Class implemented logistic regression model with CCD optimization"""
    def __init__(self):
        pass
    
    def set_std_mean(self,X, epsilon = 1e-8):
        """Part of fitting where mean and std of variables are estimated"""
        self.mean = np.mean(X, axis = 0)
        std = np.std(X,axis = 0)
        self.std = np.where(std > 0, std, epsilon)

    def standarize(self, X):
        """Standarization of data. Used in fitting and predicting"""
        return (X-self.mean)/self.std
    
    def fit(self, X, y, max_iter=100, a = 1,weights = True, user_lambda = None, fit_intercept = True, plots = False):
        """
        Fitting model with input data using CCD algorithm.

        :param X: training data, which is later standardized
        :param y: target variable of the training data
        :param max_iter: number of updates for each coefficient
        :param a: elastic_net parameter (a * lasso + (1 - a) * ridge)
        :param weights: if True, uses a version with weighted observations; otherwise, weight = 1/n
        :param user_lambda: if None, performs warm-up training as described in the article, starting with the smallest lambda for which all coefficients are zero, and then decreasing to a value 1000 times smaller. Otherwise, uses the given lambda value for max_iter iterations.
        :param fit_intercept: if False, B0 = 0; if True, B0 satisfies the equation sigmoid(B0) = prior. Recommended for unbalanced data.
        :param plots: if True, displays plots related to the change in the loss function and coefficients over iterations.
        """
        X= np.array(X)
        n, p = X.shape
        y = np.array(y)
        self.set_std_mean(X)
        X = self.standarize(X)
        self.B= np.zeros(p)
        q=1/n
        wx2 = 1
        z = q
        log_likes=[]
        coefs = np.zeros((max_iter,p))
        i=0
        if fit_intercept:
            prior = y.mean()
        else:
            prior= 0.5
        self.B0 = np.log(prior/(1-prior))
        if weights:
            z = prior*(1-prior)
        if user_lambda == None: 
            lambda_max= np.max(np.abs((y- prior)@X*z)) #since B = 0 
            if a != 0:
                lambda_max /= a
            lambdas = np.logspace(np.log10(lambda_max), np.log10(0.001*lambda_max), max_iter)
        else:
            lambdas = np.repeat(user_lambda, max_iter)

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
                self.B[j] = soft_thresholding(sum[0],lambd*a)/(wx2 +lambd*(1-a) + 1e-8)
            if plots:
                preds = sigmoid(X@self.B+ self.B0)
                log_like = -(y*np.log(preds)+(1-y)*np.log(1-preds)).sum()
                log_likes.append(log_like)
                coefs[i:,]=self.B
                i+=1
        if plots:
            iters = np.arange(0,max_iter)
            plt.figure()
            plt.plot(iters, log_likes)
            plt.xlabel('Iteration')
            plt.ylabel("Loss")
            plt.title("Change of loss function through iterations")
            plt.show()

            
            coefs = coefs.T  # Transpose to have features as rows
            
            for coef, feature in zip(coefs, range(X.shape[1])):
                plt.plot(iters, coef, label=f'Feature {feature}')

            plt.xlabel('Iteration')
            plt.ylabel('Coefficients')
            plt.title('Change of coefficients through iterations')
            plt.legend()
            plt.show()


        
    def predict_proba(self,X):
        """Return vector of propabilities of class 1 for each observation"""
        X = self.standarize(X)
        return sigmoid(X@self.B+ self.B0) 
    
    def predict(self, X):
        """Return vector of predictions using threshold 0.5 for propability"""
        X = self.standarize(X)
        return np.round(sigmoid(X@self.B + self.B0))

    def validate(self, X_valid, y_valid, measure):
        """
        Compute measures for validation data set

        Possible metrics:
        recall, precision, roc(ROC AUC), prc(precision recall curve), balanced accuracy
        """
        y_true = np.array(y_valid)
        y_scores = self.predict_proba(X_valid)
        y_pred = (y_scores >= 0.5).astype(int)
        true_pos = np.sum((y_true == 1) & (y_pred == 1))
        false_neg = np.sum((y_true == 1) & (y_pred == 0))
        false_pos = np.sum((y_true == 0) & (y_pred == 1))
        true_neg = np.sum((y_true == 0) & (y_pred == 0))
        if measure == 'recall':
            return true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        elif measure == 'precision':
            return true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        elif measure == 'roc':
            return self.ROC_AUC(y_true, y_scores)
        elif measure == 'prc':
            return self.PR_AUC(y_true, y_scores)
        elif measure == 'F-score':
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif measure == 'balanced accuracy':
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            spec = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
            return (recall + spec) / 2
        return 0

    def ROC_AUC(self, y_true, y_scores):
        """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores"""
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        thresholds = np.sort(np.unique(y_scores))[::-1]
        positive = np.sum(y_true == 1)
        negative = np.sum(y_true == 0)
        predictions = (y_scores[None, :] >= thresholds[:, None]).astype(int)
        true_pos = np.sum((predictions == 1) & (y_true[None, :] == 1), axis=1)
        false_pos = np.sum((predictions == 1) & (y_true[None, :] == 0), axis=1)
        true_pos_rate = true_pos / positive if positive > 0 else np.zeros_like(true_pos)
        false_pos_rate = false_pos / negative if negative > 0 else np.zeros_like(false_pos)
        idx = np.argsort(false_pos_rate)
        false_pos_rate = false_pos_rate[idx]
        true_pos_rate = true_pos_rate[idx]

        return np.trapz(true_pos_rate, false_pos_rate)

    def PR_AUC(self, y_true, y_scores):
        """Compute precision-recall pairs for different probability thresholds"""
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        thresholds = np.sort(np.unique(y_scores))[::-1]
        predictions = (y_scores[None, :] >= thresholds[:, None]).astype(int)
        true_pos = np.sum((predictions == 1) & (y_true[None, :] == 1), axis=1)
        false_pos = np.sum((predictions == 1) & (y_true[None, :] == 0), axis=1)
        false_neg = np.sum((predictions == 0) & (y_true[None, :] == 1), axis=1)
        denom_prec = true_pos + false_pos
        precision = np.divide(true_pos, denom_prec, out=np.ones_like(true_pos, dtype=float), where=denom_prec != 0)
        denom_rec = true_pos + false_neg
        recall = np.divide(true_pos, denom_rec, out=np.zeros_like(true_pos, dtype=float), where=denom_rec != 0)
        idx = np.argsort(recall)
        recall = recall[idx]
        precision = precision[idx]

        return np.trapz(precision, recall)
    
    def plot(self, measure, X, y, lambda_max=None, max_iter=200, lambda_num = 100, lambda_scale = 0.001,filename=None, weights = False):
        """
        Generate a plot visualizing the change in a given metric measured on the validation set, 
        predicted by a model fitted on the training set with different lambda values.

        :param measure: one of the metrics described in the validate method
        :param X, y: input data, which is automatically split into train/validation datasets 
                    used later for fitting and validating the model
        :param lambda_max: the maximum lambda value to be tested. If None, automatically determines 
                        the smallest lambda for which all coefficients equal 0.
        :param max_iter: number of iterations for the fit method
        :param lambda_num: number of lambda values to be tested 
        :param lambda_scale: the minimum lambda is determined as lambda_scale * lambda_max. 
                            The remaining lambda values are chosen sequentially from the range 
                            (lambda_max, lambda_scale * lambda_max).
        :param filename: name under which the plot will be saved in the current folder
        :param weights: determines whether the model will be fitted during testing with weighted 
                        observations or as 1/n.
        """

        X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        # Automated finding the smalest lambda with which every coefficient is 0
        if lambda_max == None:
            n = X.shape[0]
            z = 1 / n
            prior = y.mean()
            if weights:
                z = prior*(1-prior)
            self.set_std_mean(X)
            X_c = self.standarize(X)
            lambda_max = np.max(np.abs((y - prior) @ X_c * z))
        lambdas = np.linspace(lambda_max, lambda_scale * lambda_max, num=lambda_num)
        results = []
        for lambd in lambdas:
            self.fit(X, y, max_iter=max_iter, user_lambda=lambd, weights= weights)
            results.append(self.validate(X_val, y_val, measure))
    
        plt.figure()
        plt.plot(lambdas, results)
        plt.xlabel('lambda')
        plt.ylabel(measure)
        plt.title(f"Change of {measure} through lambdas with CCD Logistic Regression")
        if filename is not None:
            plt.savefig(filename)
        plt.show()
        best_index = np.argmax(results)
        best_lambda = self.lambdas_list[best_index]
        return best_lambda
    
    def plot_coefficients(self, X, y, lambda_max=None, max_iter=200, lambda_num = 100, lambda_scale = 0.001,filename=None, weights = False):
        """
        Generate a plot visualizing the change in coefficients for different lambda values.

        :param X, y: input data used for fitting the model
        :param lambda_max: the maximum lambda value to be tested. If None, automatically determines 
                        the smallest lambda for which all coefficients equal 0.
        :param max_iter: number of iterations for the fit method
        :param lambda_num: number of lambda values to be tested 
        :param lambda_scale: the minimum lambda is determined as lambda_scale * lambda_max. 
                            The remaining lambda values are chosen sequentially from the range 
                            (lambda_max, lambda_scale * lambda_max).
        :param filename: name under which the plot will be saved in the current folder
        :param weights: determines whether the model will be fitted during testing with weighted 
                        observations or as 1/n.
        """

        # Automated finding the smallest lambda with which every coefficient is 0
        if lambda_max== None:
            n = X.shape[0]
            z = 1 / n
            prior = y.mean()
            self.set_std_mean(X)
            X_c = self.standarize(X)
            if weights:
                z = prior*(1-prior)
            lambda_max = np.max(np.abs((y - prior) @ X_c * z))
        lambdas = np.linspace(lambda_max, lambda_scale * lambda_max, num=lambda_num)
        results = []
        for lambd in lambdas:
            self.fit(X, y, max_iter=max_iter, user_lambda=lambd, weights= weights)
            results.append(self.B)
    
        coefs = np.array(results).T  # Transpose to have features as rows

        for coef, feature in zip(coefs, range(X.shape[1])):
            plt.plot(lambdas, coef, label=f'Feature {feature}')
 
        plt.xlabel('Lambda')
        plt.ylabel('Coefficients')
        plt.title('Coefficients for different lambdas with CCD Logistic Regression')
        plt.legend()
        if filename is not None:
            plt.savefig(filename)
        plt.show()
    