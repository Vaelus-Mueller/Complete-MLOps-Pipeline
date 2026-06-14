import yaml
import copy
import os
from pathlib import Path

from model_training import train


def load_base_config(path='configs/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def write_config(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f)


def main():
    base = load_base_config()
    experiments = []

    # Create 5 variants: 3 logistic with different C, 2 random forest with different estimators
    variants = [
        {'model_type': 'logistic_regression', 'model_params': {'C': 0.01, 'max_iter': 200, 'solver': 'lbfgs', 'class_weight': 'balanced'}},
        {'model_type': 'logistic_regression', 'model_params': {'C': 0.1, 'max_iter': 200, 'solver': 'lbfgs', 'class_weight': 'balanced'}},
        {'model_type': 'logistic_regression', 'model_params': {'C': 1.0, 'max_iter': 200, 'solver': 'lbfgs', 'class_weight': 'balanced'}},
        {'model_type': 'random_forest', 'model_params': {'n_estimators': 50, 'random_state': base['training']['random_state'], 'class_weight': 'balanced'}},
        {'model_type': 'random_forest', 'model_params': {'n_estimators': 100, 'random_state': base['training']['random_state'], 'class_weight': 'balanced'}},
    ]

    base_dir = Path('configs')
    for i, v in enumerate(variants, start=1):
        cfg = copy.deepcopy(base)
        cfg['training']['model_type'] = v['model_type']
        cfg['training']['model_params'] = v['model_params']
        cfg_path = base_dir / f'config_experiment_{i}.yaml'
        write_config(cfg, str(cfg_path))
        print(f'Running experiment {i}: {v}')
        try:
            train(str(cfg_path))
        except Exception as e:
            print(f'Experiment {i} failed: {e}')


if __name__ == '__main__':
    main()
