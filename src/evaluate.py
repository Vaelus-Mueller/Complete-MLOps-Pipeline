import yaml

try:
    from src.data_preprocessing import load_data, split_features_target
except ImportError:
    from data_preprocessing import load_data, split_features_target


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_dataset(config_path: str):
    config = load_config(config_path)
    df = load_data(config['data']['raw_path'])
    X, y = split_features_target(df, config['data']['target_column'])
    return df, X, y


if __name__ == '__main__':
    df, X, y = evaluate_dataset('configs/config.yaml')
    print('Dataset shape:', df.shape)
    print('Target distribution:')
    print(y.value_counts(normalize=True))
