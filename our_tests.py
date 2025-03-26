from Implementation import logisitic_regression
import openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def compare_with_logistic_l1(X_train, y_train, X_test, y_test, max_count=None, max_iter=100):
    if not max_count:
        max_count = float('inf')
    
    model_own = logisitic_regression()
    model_own.fit(X_train, y_train,a=1.0, epsilon=0.001, K=50, weights=False, lambdas=None, max_count=max_count)

    y_pred_own = model_own.predict(X_test)
    accuracy_own = accuracy_score(y_test, y_pred_own)
    print(f"Test accuracy for our model l1: {accuracy_own:.4f}")

    model_sklearn = LogisticRegression(penalty='l1',solver='liblinear', C=1.0, max_iter=max_iter, random_state=42)
    model_sklearn.fit(X_train, y_train)
    y_pred_sklearn = model_sklearn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Test accuracy for sklearn model l1: {accuracy_sklearn:.4f}")