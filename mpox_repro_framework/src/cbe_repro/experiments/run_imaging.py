# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, pathlib, yaml, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from cbe_repro.synth.image_loader import ImageFolderDataset

def set_seed(seed: int):
    np.random.seed(seed)

def load_cfg(name: str):
    cfg_dir = pathlib.Path(__file__).resolve().parents[1] / "configs"
    with open(cfg_dir / name, "r") as f:
        return yaml.safe_load(f)

def _load_dataset_entry(reg_name: str):
    cfg_dir = pathlib.Path(__file__).resolve().parents[1] / "configs"
    with open(cfg_dir / "datasets.yaml","r") as f:
        reg = yaml.safe_load(f)
    return reg[reg_name]

def main(config_name="imaging_baseline.yaml"):
    cfg = load_cfg(config_name)
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    ds_reg = _load_dataset_entry(cfg["dataset"])
    root = ds_reg["root"]
    pos = ds_reg["classes"]["positive"]
    neg = ds_reg["classes"]["negative"]
    splits = ds_reg["splits"]

    # Train features (optionally with deterministic oversampling)
    train_ds = ImageFolderDataset(root, splits["train"], pos_cls=pos, neg_cls=neg,
                                  seed=seed, img_size=cfg["image_size"])
    X_tr, y_tr = train_ds.as_features_labels(
        synth_enabled=cfg.get("synth",{}).get("enabled", False),
        synth_multiplier=float(cfg.get("synth",{}).get("minority_multiplier", 1.0))
    )

    # Validation features (clean)
    val_ds = ImageFolderDataset(root, splits["val"], pos_cls=pos, neg_cls=neg,
                                seed=seed, img_size=cfg["image_size"])
    X_va, y_va = val_ds.as_features_labels(synth_enabled=False)

    # Simple, deterministic classifier
    model = LogisticRegression(max_iter=300, **cfg.get("model",{}).get("params", {}))
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)

    acc = float(accuracy_score(y_va, y_pred))
    f1  = float(f1_score(y_va, y_pred))

    manifest = {
        "run_id": str(int(time.time())),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_name": config_name,
        "metrics": {"acc": acc, "f1": f1},
        "notes": f"imaging pipeline; synth={cfg.get('synth',{}).get('enabled', False)}; img_size={cfg['image_size']}"
    }
    out_dir = pathlib.Path.cwd() / "runs" / manifest["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="imaging_baseline.yaml")
    args = ap.parse_args()
    main(config_name=args.config_name)
