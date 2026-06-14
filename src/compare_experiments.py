import os
import mlflow
import pandas as pd


def find_best_run(experiment_name: str, primary_metric: str = 'f1_score'):
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE', 'true')
    mlflow.set_tracking_uri('mlruns')
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f'Experiment not found: {experiment_name}')
    runs = mlflow.search_runs([experiment.experiment_id], order_by=[f'metrics.{primary_metric} DESC'])
    if runs.empty:
        raise ValueError('No runs found for experiment')
    best_run = runs.iloc[0]
    return best_run


def print_best_run(experiment_name: str, primary_metric: str = 'f1_score'):
    best = find_best_run(experiment_name, primary_metric)
    print(f"Best run for {experiment_name} by {primary_metric}:")
    print(best[['run_id', 'metrics.accuracy', 'metrics.f1_score', 'metrics.roc_auc', 'params.model_type', 'params.C']])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare MLflow experiments and find best run')
    parser.add_argument('--experiment', default='adult_income_experiment')
    parser.add_argument('--metric', default='f1_score')
    args = parser.parse_args()

    print_best_run(args.experiment, args.metric)
