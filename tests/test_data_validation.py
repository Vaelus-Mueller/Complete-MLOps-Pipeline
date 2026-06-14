import pandas as pd
from src.data_preprocessing import load_data


def test_expected_columns_present():
    df = load_data('data/adult.csv')
    expected = {'age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income'}
    assert expected.issubset(set(df.columns))


def test_target_values_expected():
    df = load_data('data/adult.csv')
    assert set(df['income'].dropna().unique()) <= {'<=50K', '>50K'}


def test_numeric_ranges():
    df = load_data('data/adult.csv')
    assert df['age'].between(17, 90).all()
    assert df['hours-per-week'].between(1, 99).all()
    assert df['education-num'].between(1, 16).all()
