# Baseline vs GenAI-assisted Reproducibility (Latest Runs)

## Per-config latest metrics
| config                |      acc |       f1 | started             |     run_id |
|:----------------------|---------:|---------:|:--------------------|-----------:|
| imaging_baseline.yaml | 0.595745 | 0.577778 | 2025-09-04 11:53:46 | 1756979626 |
| imaging_genai.yaml    | 0.574468 | 0.6      | 2025-09-04 11:53:47 | 1756979627 |

## Baseline vs GenAI (delta)
| task    |   baseline_acc |   genai_acc |   delta_acc |   baseline_f1 |   genai_f1 |   delta_f1 |
|:--------|---------------:|------------:|------------:|--------------:|-----------:|-----------:|
| imaging |       0.595745 |    0.574468 |  -0.0212766 |      0.577778 |        0.6 |  0.0222222 |