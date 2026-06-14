import os
import yaml
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_drift_monitoring(config_path: str):
    config = load_config(config_path)

    reference = pd.read_csv(config['data']['reference_path'])
    production = pd.read_csv(config['data']['production_path'])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=production)

    os.makedirs(os.path.dirname(config['data']['drift_report_path']), exist_ok=True)
    report.save_html(config['data']['drift_report_path'])
    print(f'Drift report saved to {config["data"]["drift_report_path"]}')

    result = report.as_dict()['metrics'][0]['result']
    drift_share = result['share_of_drifted_columns']
    drifted = result['drift_by_columns']

    print('Drift summary:')
    print(f' - total_features: {result["number_of_columns"]}')
    print(f' - drifted_features: {result["number_of_drifted_columns"]}')
    print(f' - drift_share: {drift_share:.3f}')
    print(' - dataset_drift:', result['dataset_drift'])

    if drifted:
        drifted_features = [name for name, info in drifted.items() if info.get('drift_detected')]
        print(' - drifted_feature_names:', drifted_features)

    if drift_share >= config['data']['drift_threshold']:
        raise RuntimeError(f'Drift share {drift_share:.3f} exceeds threshold {config["data"]["drift_threshold"]}')

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Monitor data drift using Evidently')
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    run_drift_monitoring(args.config)
