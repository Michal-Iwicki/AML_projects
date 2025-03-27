from Implementation import logisitic_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def fit_our_model(X_train, y_train, X_test, y_test):  
    model_own = logisitic_regression()
    model_own.fit(X_train, y_train,a=1.0, weights=False, lambdas=None)

    y_pred_own = model_own.predict(X_test)
    accuracy_own = accuracy_score(y_test, y_pred_own)
    print(f"Test accuracy for our model l1: {accuracy_own:.4f}")
    return y_pred_own

def fit_sklearn_l1(X_train, y_train, X_test, y_test, max_iter=100):
    model_sklearn = LogisticRegression(penalty='l1',solver='liblinear', C=1.0, max_iter=max_iter, random_state=42)
    model_sklearn.fit(X_train, y_train)
    y_pred_sklearn = model_sklearn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Test accuracy for sklearn model l1: {accuracy_sklearn:.4f}")
    return y_pred_sklearn
    
def fit_sklearn_no_penalty(X_train, y_train, X_test, y_test, max_iter=100):
    model_sklearn = LogisticRegression(penalty=None, C=1.0, max_iter=max_iter, random_state=42)
    model_sklearn.fit(X_train, y_train)
    y_pred_sklearn = model_sklearn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Test accuracy for sklearn model l1: {accuracy_sklearn:.4f}")
    return y_pred_sklearn