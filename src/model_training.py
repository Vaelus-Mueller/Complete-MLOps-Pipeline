import os
import joblib
import mlflow
import yaml
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    from src.data_preprocessing import load_data, split_features_target, MissingValueImputer
except ImportError:
    from data_preprocessing import load_data, split_features_target, MissingValueImputer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model(model_type: str, model_params: dict):
    if model_type == 'logistic_regression':
        return LogisticRegression(**model_params)
    elif model_type == 'random_forest':
        return RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def build_pipeline(config):
    numeric_features = config['training']['numeric_features']
    categorical_features = config['training']['categorical_features']

    numeric_transformer = Pipeline(steps=[
        ('imputer', MissingValueImputer(numeric_strategy='median', categorical_strategy='constant')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', MissingValueImputer(numeric_strategy='median', categorical_strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = build_model(config['training']['model_type'], config['training']['model_params'])

    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])


def evaluate_model(model, X_test, y_test, config):
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions, pos_label='>50K' if y_test.dtype == 'object' else 1),
    }
    if proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test.map({'<=50K': 0, '>50K': 1}) if y_test.dtype == 'object' else y_test, proba)
    else:
        metrics['roc_auc'] = 0.0
    return metrics


def save_model(model, path):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    joblib.dump(model, path)


def log_mlflow_run(config, run_metrics, data_hash):
    mlflow.set_tracking_uri(config['logging']['mlflow_tracking_uri'])
    mlflow.set_experiment(config['logging']['experiment_name'])
    with mlflow.start_run() as run:
        mlflow.log_params(config['training']['model_params'])
        mlflow.log_param('model_type', config['training']['model_type'])
        mlflow.log_param('data_uri', config['data']['raw_path'])
        mlflow.log_param('data_hash', data_hash)
        for metric_name, metric_value in run_metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))
        mlflow.log_artifact(config['data']['output_model_path'])
        return run.info.run_id


def compute_data_hash(df: pd.DataFrame) -> str:
    return str(hash(tuple(df.columns.tolist())) + hash(tuple(df.head(5).to_numpy().flatten())))


def train(config_path: str):
    config = load_config(config_path)
    df = load_data(config['data']['raw_path'])
    X, y = split_features_target(df, config['data']['target_column'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['training']['test_size'], random_state=config['training']['random_state'], stratify=y)

    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test, config)
    save_model(pipeline, config['data']['output_model_path'])

    data_hash = compute_data_hash(df)
    run_id = log_mlflow_run(config, metrics, data_hash)

    print('Training completed. Metrics:')
    for name, value in metrics.items():
        print(f' - {name}: {value:.4f}')
    print(f'MLflow run id: {run_id}')

    thresholds = config['metrics']['thresholds']
    if metrics['f1_score'] < thresholds['f1_score']:
        raise RuntimeError(f"Primary metric f1_score below threshold: {metrics['f1_score']:.4f} < {thresholds['f1_score']}")

    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train model with config.')
    parser.add_argument('--config', default='configs/config.yaml', help='Path to YAML config file')
    args = parser.parse_args()
    train(args.config)
