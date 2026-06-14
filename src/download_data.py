import io
import os
import requests
import pandas as pd


def download_adult_dataset(output_path: str):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(
        io.StringIO(response.text),
        header=None,
        names=columns,
        na_values=' ?',
        skipinitialspace=True
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def create_reference_and_production_datasets(raw_path: str, reference_path: str, production_path: str, random_state: int = 42):
    df = pd.read_csv(raw_path)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    reference = df.iloc[:20000].copy()
    production = df.iloc[20000:].copy()

    production['hours-per-week'] = production['hours-per-week'].clip(upper=99)
    production.loc[production.sample(frac=0.1, random_state=random_state).index, 'native-country'] = 'Mexico'

    os.makedirs(os.path.dirname(reference_path), exist_ok=True)
    os.makedirs(os.path.dirname(production_path), exist_ok=True)
    reference.to_csv(reference_path, index=False)
    production.to_csv(production_path, index=False)
    return reference, production


if __name__ == '__main__':
    raw_path = 'data/adult.csv'
    reference_path = 'data/adult_reference.csv'
    production_path = 'data/adult_production.csv'

    print(f'Downloading Adult dataset to {raw_path}...')
    download_adult_dataset(raw_path)
    print('Creating reference and production files...')
    create_reference_and_production_datasets(raw_path, reference_path, production_path)
    print('Data preparation complete.')
