from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder

def add_permuted_columns(df):
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
    print(f'Unique values of label column {pd.unique(df[target_column])}')
    df['target'] = (df[target_column].isin(positive_class)).astype(int)
    df.drop(columns=[target_column], inplace=True)
    print(df["target"].value_counts())
    
def fill_na(df):
    """
    fills null values with median
    """
    na_counts = df.isnull().sum()
    na_columns = na_counts[na_counts > 0]
    for column, count in na_columns.items():
        print(f"{column} {count} null values")
    df.fillna(df.median(), inplace=True)
    
    
def delete_corr_columns(df):
    corr_matrix = df.drop(columns=['target']).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    print("deleted features:", to_drop)
    df.drop(columns=to_drop, inplace=True)
    print(f"Shape after deletion {df.shape}")
    
    
def check_feature_sample_ratio(df):
    print(f'The number of features is enough: {len(df.columns) * 2 > len(df)}, number of features: {len(df.columns)}')
    
def split(df):
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X, y.astype(int), X_train, X_test, y_train.astype(int), y_test.astype(int)

def one_hot_encode_categorical(df):
    feature_cols = df.columns.drop("target")
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            
def drop_categorical(df):
    target = "target"
    columns_to_keep = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) or col == target]
    print(f"Number of columns to keep {len(columns_to_keep)}, to delete {len(df.columns) - len(columns_to_keep)}")
    return df[columns_to_keep]