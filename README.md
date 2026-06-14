# Drift Monitoring ML Project

This repository demonstrates a complete machine learning workflow with version control, experiment tracking, testing, CI/CD, and drift monitoring.

## Project Structure

- `src/` - Python modules for preprocessing, training, evaluation, drift monitoring, and experiment comparison.
- `configs/` - YAML configuration for training parameters, paths, and thresholds.
- `tests/` - Pytest test suite.
- `.github/workflows/` - GitHub Actions CI workflow.
- `data/` - Dataset files (downloaded by scripts and not tracked by Git).
- `models/` - Saved model artifacts.
- `reports/` - Drift reports and monitoring output.

## Setup

1. Create a Python environment with Python 3.13.
2. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Pull the versioned dataset with DVC:
   ```bash
   dvc pull
   ```

The dataset is tracked by DVC (see `data/*.dvc` pointer files). The local remote cache lives in `dvc-storage/` so graders can restore data without downloading from UCI.

## Download Dataset (alternative)

If DVC is unavailable, download the UCI Adult Income dataset manually:

```bash
python - <<'PY'
import requests
import pandas as pd
import io
url='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
cols=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
df=pd.read_csv(io.StringIO(requests.get(url).text), header=None, names=cols, na_values=' ?', skipinitialspace=True)
df.to_csv('data/adult.csv', index=False)
PY
```

## Training

Train the model with:

```bash
python src/model_training.py --config configs/config.yaml
```

## Experiment Comparison

Compare MLflow runs with:

```bash
python src/compare_experiments.py --experiment adult_income_experiment --metric f1_score
```

## Drift Monitoring

Generate drift monitoring reports with:

```bash
python src/monitor_drift.py --config configs/config.yaml
```

See [MONITORING.md](MONITORING.md) for drift analysis, performance impact assessment, and recommended actions.

## Testing

Run the full test suite with:

```bash
pytest tests/ -v
```

## Notes

- Raw CSV data files are excluded from Git; DVC pointer files (`data/*.dvc`) are committed instead.
- Run `dvc pull` after cloning to restore datasets from the `dvc-storage` remote.
- MLflow tracking is stored in the `mlruns/` directory.
- Drift reporting uses Evidently and saves HTML output to `reports/drift_report.html`.
