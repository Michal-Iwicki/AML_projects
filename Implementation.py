import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import measures

def soft_thresholding(a, b):
    return np.sign(a) * np.maximum(np.abs(a) - b, 0)

def sigmoid(x):
    x = np.clip(x, -500, 500)
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
    
    def fit(self, X, y, max_iter=100, a = 1,weights = True, user_lambda = None, fit_intercept = True, X_valid=None, y_valid=None, measure=None):
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
            lambdas = np.repeat(user_lambda, max_iter)

        self.lambdas_list = []
        self.coeff_list = []
        self.inter_list = []
        self.validations = []
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
            self.lambdas_list.append(lambd)
            self.coeff_list.append(np.copy(self.B))
            self.inter_list.append(self.B0)
            
            if X_valid is not None and y_valid is not None and measure is not None:
                X_valid_standardised = self.standarize(X_valid)
                proba_valid = sigmoid(X_valid_standardised @ self.B + self.B0)
                score_valid = self.evaluate(y_valid, proba_valid, measure)
                self.validations.append(score_valid)
        if len(self.validations) > 0:
            best_index = np.argmax(self.validations)
            self.B = self.coeff_list[best_index]
            self.B0 = self.inter_list[best_index]
            self.best_lambda = self.lambdas_list[best_index]
        
    def predict_proba(self,X):
        X = self.standarize(X)
        return sigmoid(X@self.B+ self.B0) 
    
    def validate(self, X_valid, y_valid, measure):
        X_valid = self.standarize(X_valid)
        scores_valid = []
        for B, B0 in zip(self.coeff_list, self.inter_list):
            proba_valid = sigmoid(X_valid @ B + B0)
            score_valid = self.evaluate(y_valid, proba_valid, measure)
            scores_valid.append(score_valid)
        return np.array(scores_valid)
    
    def predict(self, X):
        X = self.standarize(X)
        return np.round(sigmoid(X@self.B + self.B0))

    # def validate(self, X_valid, y_valid, measure):
    #     return measure(self.fit(X_valid), y_valid)
    #
    # def plot(self, measure, X, y, lambdas=None, max_iter=200):
    #     metrics = {
    #         "precision": measure.precision,
    #         "recall": measure.recall,
    #         "f_measure": measure.f_measure,
    #         "balanced_accuracy": measure.balanced_accuracy,
    #         "auc_roc": measure.auc_roc,
    #         "auc_pr": measure.auc_pr
    #     }
    #     metric = metrics.get(measure)
    #     X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    #     # just for testing if works can delete it
    #     if not lambdas:
    #         n = X.shape[0]
    #         z = 1 / n
    #         prior = y.mean()
    #         lambda_max = np.max(np.abs((y - prior) @ X * z))
    #         lambdas = np.linspace(lambda_max - 0.001, 0.001 * lambda_max, num=100)
    #     results = []
    #     for lambd in lambdas:
    #         self.fit(X, y, max_iter=max_iter, user_lambda=lambd)
    #         results.append(self.validate(X_val, y_val, metric))
    #
    #     # Plot to be done
    #
    # def plot_coefficients(self, X, y, lambdas=None, max_iter=200):
    #     if not lambdas:
    #         n = X.shape[0]
    #         z = 1 / n
    #         prior = y.mean()
    #         lambda_max = np.max(np.abs((y - prior) @ X * z))
    #         lambdas = np.linspace(lambda_max - 0.001, 0.001 * lambda_max, num=100)
    #     results = []
    #     for lambd in lambdas:
    #         self.fit(X, y, max_iter=max_iter, user_lambda=lambd)
    #         results.append(self.B)
    #
    #     # Plot to be done

    def evaluate(self, y_true, y_scores, measure):
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
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

    def plot(self, X_valid, y_valid, measure, filename=None):
        scores = self.validate(X_valid, y_valid, measure)
        best_index = np.argmax(scores)
        best_lambda = self.lambdas_list[best_index]
        plt.figure()
        plt.plot(self.lambdas_list, scores)
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel(measure)
        plt.title(f"{measure} vs lambda")
        if filename is not None:
            plt.savefig(filename)
        plt.show()
        return best_lambda

    def plot_coefficients(self, filename=None):
        plt.figure()
        coefs = np.array(self.coeff_list)
        for j in range(coefs.shape[1]):
            plt.plot(self.lambdas_list, coefs[:, j])
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('Coefficients')
        plt.title('Coefficients vs Lambda')
        if filename is not None:
            plt.savefig(filename)
        plt.show()