# Drift Monitoring Analysis

This document summarizes data drift findings from comparing the reference dataset (`data/adult_reference.csv`) against the simulated production dataset (`data/adult_production.csv`) using Evidently's `DataDriftPreset`.

## Methodology

- **Reference data**: First 20,000 rows sampled from the UCI Adult Income dataset (stable baseline).
- **Production data**: Remaining rows with intentional distribution shifts injected to simulate real-world drift:
  - 10% of production rows have `native-country` changed to `Mexico`.
  - `hours-per-week` values are clipped at 99.
- **Tooling**: `src/monitor_drift.py` runs Evidently drift detection on all 15 features and writes an HTML report to `reports/drift_report.html`.
- **Alert threshold**: Drift share ≥ 0.25 (configured in `configs/config.yaml` as `data.drift_threshold`).

## Drift Findings

| Feature | Drifted? | Why |
|---------|----------|-----|
| `native-country` | **Yes** | Production data over-represents `Mexico` (10% of rows relabeled) compared to the reference distribution, causing a categorical distribution shift. |
| `hours-per-week` | No | Clipping at 99 affects only extreme outliers; the overall distribution remains similar enough to stay below the drift threshold. |
| All other features (age, workclass, education, etc.) | No | Sampled from the same underlying dataset without modification. |

**Summary metrics** (latest run):

- Total features monitored: 15
- Drifted features: 1 (`native-country`)
- Drift share: ~0.067 (6.7%)
- Dataset-level drift: **No** (below 0.25 threshold)

## Performance Impact

The income prediction model relies heavily on categorical features including `native-country`. A sustained shift in country distribution can:

1. **Reduce recall** for income classes underrepresented in the new geography mix.
2. **Skew probability calibration** because the model was trained on the original country distribution.
3. **Increase false positives/negatives** for edge-case demographic segments.

At the current drift level (single feature, ~7% share), performance degradation is expected to be **minor**. If `native-country` drift persists or expands to additional features, model accuracy and F1 score could drop below the configured training thresholds (F1 ≥ 0.65).

## Recommended Actions

1. **Monitor `native-country` weekly** — track its drift score trend; retrain if drift share exceeds 0.25.
2. **Collect fresh labeled production data** from the shifted population and add it to the training set via DVC versioning.
3. **Retrain the model** with updated reference data once drift is confirmed persistent (run `python src/model_training.py --config configs/config.yaml`).
4. **Review the HTML report** at `reports/drift_report.html` for per-feature statistical test details and visual comparisons.

## Running Drift Monitoring

```bash
python src/monitor_drift.py --config configs/config.yaml
```

The script prints a drift summary to stdout, saves the HTML report, and exits with code 1 if the drift share exceeds the configured threshold.
