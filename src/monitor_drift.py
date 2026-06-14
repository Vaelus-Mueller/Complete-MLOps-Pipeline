import os
import re
import yaml
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _parse_drift_snapshot(snapshot):
    metrics = snapshot.dict()['metrics']
    drift_count_metric = next(
        (metric for metric in metrics if 'DriftedColumnsCount' in metric['metric_name']),
        None,
    )
    value_drifts = [metric for metric in metrics if metric['metric_name'].startswith('ValueDrift(')]

    drift_share = float(drift_count_metric['value']['share']) if drift_count_metric else 0.0
    drifted_features = drift_count_metric['value']['count'] if drift_count_metric else 0
    total_features = len(value_drifts)
    dataset_drift = drift_share >= 0.5

    drifted_feature_names = []
    for metric in value_drifts:
        column_match = re.search(r'column=([^,]+)', metric['metric_name'])
        threshold_match = re.search(r'threshold=([^)]+)\)', metric['metric_name'])
        if not column_match or not threshold_match:
            continue
        column = column_match.group(1)
        threshold = float(threshold_match.group(1))
        drift_score = float(metric['value'])
        if drift_score >= threshold:
            drifted_feature_names.append(column)

    return {
        'number_of_columns': total_features,
        'number_of_drifted_columns': int(drifted_features),
        'share_of_drifted_columns': drift_share,
        'dataset_drift': dataset_drift,
        'drifted_feature_names': drifted_feature_names,
    }


def run_drift_monitoring(config_path: str):
    config = load_config(config_path)

    reference = pd.read_csv(config['data']['reference_path'])
    production = pd.read_csv(config['data']['production_path'])

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=production)

    os.makedirs(os.path.dirname(config['data']['drift_report_path']), exist_ok=True)
    snapshot.save_html(config['data']['drift_report_path'])
    print(f'Drift report saved to {config["data"]["drift_report_path"]}')

    result = _parse_drift_snapshot(snapshot)
    drift_share = result['share_of_drifted_columns']

    print('Drift summary:')
    print(f' - total_features: {result["number_of_columns"]}')
    print(f' - drifted_features: {result["number_of_drifted_columns"]}')
    print(f' - drift_share: {drift_share:.3f}')
    print(' - dataset_drift:', result['dataset_drift'])

    if result['drifted_feature_names']:
        print(' - drifted_feature_names:', result['drifted_feature_names'])

    if drift_share >= config['data']['drift_threshold']:
        raise RuntimeError(f'Drift share {drift_share:.3f} exceeds threshold {config["data"]["drift_threshold"]}')

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Monitor data drift using Evidently')
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    run_drift_monitoring(args.config)
