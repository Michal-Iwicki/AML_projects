from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder

def add_permuted_columns(df):
    """
    adds permuted versions of existing feature columns to the dataframe 
    to ensure that the number of features is at least half the number of rows

    :param df: input dataframe with a column named "target" containing a label and feature columns
    :returns: new dataframe with new permuted columns
    """
    current_features = df.shape[1] - 1 
    needed_features = int(np.ceil(0.5 * df.shape[0]))
    n_to_add = needed_features - current_features
    print(f"Current num of features: {current_features}, adding: {needed_features} features")

    if n_to_add > 0:
        new_cols = {} 
        original_cols = list(df.columns.drop("target"))
        n = len(original_cols)
        for i in range(n_to_add):
            col = original_cols[i % n]
            new_col = np.random.permutation(df[col].values)
            # optional for noise uncommment
            # noise = np.random.normal(0, 1e-6, size=new_col.shape)
            # new_cols[f"new_{i}"] = new_col + noise
            new_cols[f"new_{i}"] = new_col 

        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        df = df.copy()
        print("New dataframe shape:", df.shape)
        return df

def assign_target_column(df, target_column, positive_class):
    """
    creates a binary column named "target" based on the specified values representing the positive class

    :param df: dataframe containing the original target_column
    :param target_column: name of the column to be converted into binary labels
    :param positive_class: list of values that are going to be aggregated to the class labeleb "1"
    :returns: dataframe with a new 'target' column and without the original target column
    """
    print(f'Unique values of label column {pd.unique(df[target_column])}')
    df['target'] = (df[target_column].isin(positive_class)).astype(int)
    df.drop(columns=[target_column], inplace=True)
    print(df["target"].value_counts())
    
def fill_na(df):
    """
    fills null values with median
    :param df: input dataframe 
    """
    na_counts = df.isnull().sum()
    na_columns = na_counts[na_counts > 0]
    for column, count in na_columns.items():
        print(f"{column} {count} null values")
    df.fillna(df.median(), inplace=True)
    
    
def delete_corr_columns(df):
    """
    removes highly correlated feature columns 

    :param df: input dataframe with feature columns and a column named "target"
    :returns: dataframe with correlated columns removed
    """
    corr_matrix = df.drop(columns=['target']).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    print("deleted features:", to_drop)
    df.drop(columns=to_drop, inplace=True)
    print(f"Shape after deletion {df.shape}")
    
    
def check_feature_sample_ratio(df):
    """
    checks if the number of features is at least half of the number of samples and prints the result
    :param df: input dataframe with feature columns
    """
    print(f'The number of features is enough: {len(df.columns) * 2 > len(df)}, number of features: {len(df.columns)}')
    
def split(df):
    """
    splits the dataframe into train and test sets with a 70/30 ratio

    :param df: dataframe with column named "target"
    :returns: X, y, X_train,  X_test, y_train,  y_test 
    """
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42)
    return X, y.astype(int), X_train, X_test, y_train.astype(int), y_test.astype(int)
            
def drop_categorical(df):
    """
    removes all categorical columns from the dataframe except for the 'target' column
    :param df: input dataframe
    :returns: dataframe with only numeric columns and a column named "target"
    """
    target = "target"
    columns_to_keep = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) or col == target]
    print(f"Number of columns to keep {len(columns_to_keep)}, to delete {len(df.columns) - len(columns_to_keep)}")
    return df[columns_to_keep]