# 4.2.5 Generative AI–Based Interventions for Reproducibility

_Auto report generated: 2025-09-04 12:22_

## Experimental Setup
- **Imaging:** deterministic feature extractor + logistic regression; GenAI flag performs deterministic oversampling of positives.
- **Symptoms (tabular):** RandomForest; GenAI flag uses SMOTE/oversampling (tunable).
- All runs fix `seed=1337` and log a JSON manifest.

## Latest Per-Config Metrics
|     run_id | started             | config_name           | metrics                                               | notes                                       |   paper_id |   modality |   dataset_key |   model |   synth |   tuning |   ReproScore |
|-----------:|:--------------------|:----------------------|:------------------------------------------------------|:--------------------------------------------|-----------:|-----------:|--------------:|--------:|--------:|---------:|-------------:|
| 1756981362 | 2025-09-04 12:22:42 | imaging_baseline.yaml | {'acc': 0.5957446808510638, 'f1': 0.5777777777777777} | imaging pipeline; synth=False; img_size=128 |        nan |        nan |           nan |     nan |     nan |      nan |            1 |
| 1756981365 | 2025-09-04 12:22:45 | imaging_genai.yaml    | {'acc': 0.574468085106383, 'f1': 0.6}                 | imaging pipeline; synth=True; img_size=128  |        nan |        nan |           nan |     nan |     nan |      nan |            1 |

## Baseline vs GenAI (Δ)
| task    |   baseline_acc |   genai_acc |   delta_acc |   baseline_f1 |   genai_f1 |   delta_f1 |   baseline_ReproScore |   genai_ReproScore |   delta_ReproScore |
|:--------|---------------:|------------:|------------:|--------------:|-----------:|-----------:|----------------------:|-------------------:|-------------------:|
| imaging |       0.595745 |    0.574468 |  -0.0212766 |      0.577778 |        0.6 |  0.0222222 |                     1 |                  1 |                  0 |

## Reproducibility Notes
- We versioned configs, fixed RNG seeds, and saved run manifests.
- GenAI interventions were deterministic (oversampling with fixed patterns for imaging; seeded SMOTE/oversample for tabular).
- ReproScore aggregates presence of config/seed/data/manifest/augmentation control into a 0–1 score.

- **Imaging**: F1 0.578 → 0.600 (Δ +0.022); Acc 0.596 → 0.574 (Δ -0.021); ReproScore 1.00 → 1.00 (Δ +0.00).