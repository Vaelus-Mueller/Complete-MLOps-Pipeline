import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.data_preprocessing import load_data, split_features_target


def test_model_prediction_shape_and_type():
    df = load_data('data/adult.csv').sample(n=200, random_state=42)
    X, y = split_features_target(df, 'income')
    numeric = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    categorical = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric), ('cat', categorical_transformer, categorical)])
    clf = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=200, class_weight='balanced'))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    assert preds.shape == (X_test.shape[0],)
    assert preds.dtype == object


def test_model_minimum_performance():
    df = load_data('data/adult.csv').sample(n=1000, random_state=42)
    X, y = split_features_target(df, 'income')
    numeric = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    categorical = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric), ('cat', categorical_transformer, categorical)])
    clf = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=200, class_weight='balanced'))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    assert score >= 0.7
