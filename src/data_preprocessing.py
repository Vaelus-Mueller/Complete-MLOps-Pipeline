import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    return X, y


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_strategy='median', categorical_strategy='constant', fill_value='missing'):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value
        self.numeric_columns = None
        self.categorical_columns = None

    def fit(self, X, y=None):
        self.numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object', 'string']).columns.tolist()
        return self

    def transform(self, X):
        X_out = X.copy()
        if self.numeric_columns:
            if self.numeric_strategy == 'median':
                X_out[self.numeric_columns] = X_out[self.numeric_columns].fillna(X_out[self.numeric_columns].median())
            elif self.numeric_strategy == 'mean':
                X_out[self.numeric_columns] = X_out[self.numeric_columns].fillna(X_out[self.numeric_columns].mean())
            else:
                raise ValueError(f"Unsupported numeric strategy: {self.numeric_strategy}")

        if self.categorical_columns:
            if self.categorical_strategy == 'constant':
                X_out[self.categorical_columns] = X_out[self.categorical_columns].fillna(self.fill_value)
            else:
                raise ValueError(f"Unsupported categorical strategy: {self.categorical_strategy}")

        return X_out


def add_missing_flag_columns(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    df = df.copy()
    if columns is None:
        columns = df.columns.tolist()
    for col in columns:
        if col in df.columns:
            df[f'{col}_missing_flag'] = df[col].isna().astype(int)
    return df


def encode_categorical_features(df: pd.DataFrame, categorical_columns):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')
    df_out = df.copy()
    for col in categorical_columns:
        if col not in df_out.columns:
            raise ValueError(f"Categorical column '{col}' is not in dataframe")
        df_out[col] = df_out[col].astype('string').fillna('missing')
        categories = sorted(df_out[col].unique())
        mapping = {category: idx for idx, category in enumerate(categories)}
        df_out[col] = df_out[col].map(mapping)
    return df_out
