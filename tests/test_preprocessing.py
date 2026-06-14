import pandas as pd
import pytest
from src.data_preprocessing import add_missing_flag_columns, encode_categorical_features, MissingValueImputer


def test_missing_flag_columns_added():
    df = pd.DataFrame({'a': [1, None], 'b': ['x', 'y']})
    result = add_missing_flag_columns(df)
    assert 'a_missing_flag' in result.columns
    assert result['a_missing_flag'].tolist() == [0, 1]


def test_encode_categorical_features_mapping():
    df = pd.DataFrame({'color': ['red', 'blue', 'red']})
    encoded = encode_categorical_features(df, ['color'])
    assert encoded['color'].dtype == 'int64'
    assert set(encoded['color'].unique()) == {0, 1}


def test_encode_categorical_invalid_input():
    with pytest.raises(TypeError):
        encode_categorical_features('not a dataframe', ['color'])


def test_encode_categorical_missing_column():
    df = pd.DataFrame({'color': ['red']})
    with pytest.raises(ValueError):
        encode_categorical_features(df, ['nonexistent'])


def test_imputer_preserves_dataframe():
    df = pd.DataFrame({'num': [1, None], 'cat': ['a', None]})
    imputer = MissingValueImputer()
    imputer.fit(df)
    transformed = imputer.transform(df)
    assert df.isna().sum().sum() == 2
    assert transformed.isna().sum().sum() == 0


def test_imputer_unsupported_strategy():
    df = pd.DataFrame({'num': [1, None], 'cat': ['a', None]})
    imputer = MissingValueImputer(numeric_strategy='unknown')
    imputer.fit(df)
    with pytest.raises(ValueError):
        imputer.transform(df)
