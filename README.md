# GARMF-Mpox  
**GenAI-Assisted Reproducible Modelling Framework for Mpox AI Research**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)  
**Repo:** <https://github.com/chadrack-123/Framework> • **DOI:** <10.5281/zenodo.xxxxxxx> • **Appendix D:** (thesis)

## Overview
**GARMF-Mpox** is a lightweight, end-to-end framework that makes mpox AI studies **reproducible** and **auditable**. It standardises:
- **Synthetic data augmentation** (diffusion/GAN for images; SMOTE for tabular),
- **Code generation + standardisation** (runnable pipelines; pinned environments),
- **Automated documentation** (Model/Data Cards; EPIFORGE/IDMRC checklist),
- **Explainability** (Grad-CAM/Eigen/ScoreCAM for imaging; SHAP/LIME for tabular),
- **MLOps controls** (version pinning, seeds, manifests, release bundles).

The framework does **not** prescribe a single model. It provides a governed process that yields **verifiable artefacts** (configs, seeds, cards, XAI) for independent reruns.

---

## Repo structure
```
.
├─ conf/
│  └─ env/                 # Conda environment files (CPU/CUDA)
├─ src/
│  └─ cbe_repro/
│     ├─ experiments/
│     │  └─ run_unified.py            # main runner (CLI + notebook)
│     ├─ configs/
│     │  ├─ datasets.yaml             # dataset registry
│     │  ├─ papers/                   # paper profiles (YAML)
│     │  │  ├─ farzipour_2023_baseline.yaml
│     │  │  └─ ...
│     ├─ models/                      # model factories (tabular + imaging)
│     ├─ synth/                       # SMOTE / image loaders / augmentation hooks
│     └─ reporting/                   # auto-doc writers (cards + checklist)
├─ notebooks/
│  └─ Framework_runner.ipynb          # recommended entry point
├─ runs/                               # outputs per run (created automatically)
└─ artifacts/                          # release bundles (optional)
```

---

## Quick start

### 1) Create environment
```bash
# Clone
git clone https://github.com/chadrack-123/Framework
cd Framework

# Conda (CPU example; use CUDA yml if you have a GPU)
conda env create -f conf/env/environment.yml
conda activate garmf

# Install package (editable)
python -m pip install -e src
```

### 2) Point to your data (optional but recommended)
Set an environment variable so **relative** dataset paths in `configs/datasets.yaml` resolve cleanly:

**Linux/macOS**
```bash
export CBE_DATA_ROOT="/path/to/your/data-root"
```

**Windows (PowerShell)**
```powershell
$env:CBE_DATA_ROOT="C:\path\to\data-root"
```

> Tip: Keep `datasets.yaml` paths **relative** to a logical root; switch machines by updating `CBE_DATA_ROOT`.

### 3) Run a paper profile (CLI)
```bash
python -m cbe_repro.experiments.run_unified farzipour_2023_baseline.yaml
```
This creates `runs/<timestamp>/` with:
- `manifest.json` (metrics, seeds, dataset snapshot, tuning),
- `xai/` (SHAP plots for tabular; CAM overlays for imaging),
- `docs/` (Model Card, Data Card, Reproducibility Checklist),
- `used_config.json` (effective configuration).

### 4) Run via notebook (recommended)
Open `notebooks/Framework_runner.ipynb` and execute:
```python
from cbe_repro.experiments.run_unified import run_from_profile
manif = run_from_profile("farzipour_2023_baseline.yaml", return_results=True)
manif
```

---

## Configuring datasets
Edit `src/cbe_repro/configs/datasets.yaml` to register resources, e.g.:
```yaml
farzipour_symptoms:
  path: data/mpox_symptoms.csv         # resolved under CBE_DATA_ROOT (or absolute)
  label_col: status
  positive_value: mpox

azar_images:
  root: data/azar_mpox_images          # expect root/{train|val}/{positive|negative}/
  classes:
    positive: mpox
    negative: non_mpox
  image_size: 224
```

---

## Running other profiles
Profiles live in `src/cbe_repro/configs/papers/`. Each YAML defines modality, dataset key, model, synthesis, and metrics. For example:

```yaml
paper_id: farzipour_2023
modality: tabular
dataset: farzipour_symptoms
model:
  name: xgboost
  params:
    n_estimators: 300
    max_depth: 4
synth:
  enabled: true
  target_ratio: 0.5
metrics:
  seed: 1337
  ci: {enabled: true, B: 1000, alpha: 0.05}
  xai: true
  tune:
    enabled: true
    method: grid
    cv: 5
    scoring: f1
```

Run it:
```bash
python -m cbe_repro.experiments.run_unified <profile.yaml>
```

---

## Outputs you can expect
Inside `runs/<id>/`:
- `manifest.json` — metrics (Acc/F1/AUC), bootstrap CI, seeds, splits, tuned thresholds, and repository commit (if available).
- `xai/`
  - Imaging: `vit_gradcam_*.png` / `*_cam_*.png` overlays.
  - Tabular: `shap_summary.png`, `shap_top_features.png`, `shap_feature_importance.csv`.
- `docs/`
  - `model_card.md`, `data_card.md`, `repro_checklist.md`.
- `used_config.json` — exact config used (for precise reruns).

---

## Explainability (XAI)
- **Imaging:** Grad-CAM / EigenCAM / ScoreCAM on CNN/ViT targets; overlays saved under `runs/<id>/xai/`.
- **Tabular:** SHAP (Tree/Kernel as appropriate) with permutation-importance fallback; mean |SHAP| rankings and beeswarm plots saved automatically.

---

## Reproducibility controls
- **Seeds & splits** are stratified and recorded.
- **Environment** is pinned via `conf/env/environment.yml`.
- **Tuning** (grid/random) logs `best_params` and CV protocol.
- **Release bundles**: optional `artifacts/release/` with code+configs+docs for archival/DOI.

---

## Add a new paper profile
1. Register the dataset in `configs/datasets.yaml`.  
2. Create a new YAML in `configs/papers/your_paper.yaml`.  
3. Run it (CLI or notebook).  
4. Check `runs/<id>/` for artefacts; reference them in your manuscript/report.

---

## Generating thesis figures (optional)
From a specific `RUN_DIR`, you can assemble Appendix D figures:
- **Figure D.2:** CAM overlays + SHAP plots → `fig_D2_xai_panels.png`
- **Figure D.3:** Model/Data Card excerpts → `fig_D3_cards.png`
- **Figure D.4:** Checklist + `manifest.json` excerpt → `fig_D4_checklist_manifest.png`

(*See Appendix D helper cells for ready-made code snippets.*)

---

## Troubleshooting
- **`ModuleNotFoundError: cbe_repro`** → run `python -m pip install -e src` (in the repo root) or add `src/` to `PYTHONPATH` in the notebook bootstrap cell.
- **Paths not found** → set `CBE_DATA_ROOT` or use absolute paths in `datasets.yaml`.
- **CUDA/AMP errors** → use the CPU environment file or set `amp: false` in the imaging profile.
- **Grad-CAM hooks failing on in-place ReLU** → the runner switches them off; if a custom model is used, set all `nn.ReLU(inplace=False)` before CAM.

---

## Citation
If you use this framework, please cite:
```
@software{garmf_mpox_2025,
  title   = {GARMF-Mpox: GenAI-Assisted Reproducible Modelling Framework for Mpox AI Research},
  author  = Chadrack Kavula Mulamba,
  year    = {2025},
  url     = {https://github.com/chadrack-123/Framework},
  version = {v1.0},
  doi     = {10.5281/zenodo.xxxxxxx}
}
```

---

## License
MIT License — see `LICENSE` for details.

## Acknowledgements
This framework draws on best practices in XAI, MLOps, and scientific reproducibility (EPIFORGE, IDMRC). See the thesis **Appendix D** for a visual blueprint and example artefacts.
