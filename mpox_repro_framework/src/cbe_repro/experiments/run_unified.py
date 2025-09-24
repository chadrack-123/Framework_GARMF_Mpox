# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, yaml, numpy as np
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, make_scorer

from cbe_repro.models.model_zoo import ModelSpec, make_tabular, make_imaging
from cbe_repro.synth.image_loader import ImageFolderDataset
from cbe_repro.synth.symptom_smote import smote_or_oversample
from sklearn.inspection import permutation_importance

# ---- Torch / ViT imports
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ----------------------------------------------------------------------------- #
# Paths & helpers
# ----------------------------------------------------------------------------- #
ROOT    = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "src" / "cbe_repro"
CFG_DIR = PKG_DIR / "configs"
RUNS    = ROOT / "runs"

def _fail(msg: str):
    raise RuntimeError(f"[run_unified] {msg}")

def resolve_path(p: str | Path, *, expect_dir=False, expect_file=False) -> Path:
    """
    Resolve relative paths w.r.t. configs/, with optional override via CBE_DATA_ROOT.
    """
    p = Path(str(p)).expanduser()
    if p.is_absolute():
        if (expect_dir and not p.is_dir()) or (expect_file and not p.is_file()):
            _fail(f"Path does not exist: {p}")
        return p

    env_root = os.environ.get("CBE_DATA_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root) / p)
    candidates += [CFG_DIR / p, PKG_DIR / p, ROOT / p]

    for c in candidates:
        if (expect_dir and c.is_dir()) or (expect_file and c.is_file()) or (not expect_dir and not expect_file and c.exists()):
            return c
    _fail(f"Could not resolve path '{p}'. Tried: {', '.join(map(str, candidates))}")

def _yml(p: Path):
    if not p.exists():
        _fail(f"Missing YAML: {p}")
    return yaml.safe_load(p.read_text())

def _reg() -> dict:
    ds = CFG_DIR / "datasets.yaml"
    reg = _yml(ds) or {}
    if not isinstance(reg, dict):
        _fail(f"Invalid datasets.yaml format at {ds}")
    return reg

def _seed(s: int):
    s = int(s)
    np.random.seed(s)
    try:
        import random; random.seed(s)
    except Exception:
        pass
    try:
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except Exception:
        pass

def _py_counts(d: dict) -> dict:
    return {int(k): int(v) for k, v in d.items()}

# ----------------------------------------------------------------------------- #
# Torch imaging helpers
# ----------------------------------------------------------------------------- #
def _build_torch_image_datasets(entry, preprocess, seed=1337):
    """
    Uses torchvision.datasets.ImageFolder with layout:
      root/{train|val}/{positive|negative}/*.jpg
    Folder names for splits can be overridden via datasets.yaml: splits.train / splits.val.
    """
    root = resolve_path(entry["root"], expect_dir=True)
    pos = entry["classes"]["positive"]
    neg = entry["classes"]["negative"]

    splits = entry.get("splits", {}) or {}
    train_name = splits.get("train", "train")
    val_name   = splits.get("val", "val")

    ds_tr = datasets.ImageFolder(root/train_name, transform=preprocess)
    ds_va = datasets.ImageFolder(root/val_name,   transform=preprocess)

    # enforce {neg:0, pos:1}
    want = {neg: 0, pos: 1}
    if ds_tr.class_to_idx != want:
        idx_to_class = {v: k for k, v in ds_tr.class_to_idx.items()}
        def _remap_targets(ds):
            new_targets = []
            for t in ds.targets:
                clsname = idx_to_class[t]
                new_targets.append(want[clsname])
            ds.targets = new_targets
            ds.class_to_idx = want
        _remap_targets(ds_tr); _remap_targets(ds_va)

    tr_counts = dict(zip(*np.unique(np.array(ds_tr.targets), return_counts=True)))
    va_counts = dict(zip(*np.unique(np.array(ds_va.targets), return_counts=True)))
    return ds_tr, ds_va, tr_counts, va_counts

@torch.no_grad()
def _eval_torch_cls(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        prob1 = torch.softmax(logits, dim=1)[:, 1]
        all_y.append(y.cpu()); all_p.append(prob1.cpu())
    y_true = torch.cat(all_y).numpy()
    p1 = torch.cat(all_p).numpy()
    y_pred = (p1 >= 0.5).astype(int)
    return y_true, y_pred, p1

def _train_vit(model, ds_tr, ds_va, *, epochs=30, batch_size=16, lr=3e-4, weight_decay=0.05,
               cosine=True, patience=7, amp=True, seed=1337, use_genai_balance=False, linear_probe_epochs=5):
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

    torch.manual_seed(seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = (device.type == "cuda")
    pin_mem = is_cuda

    model = model.to(device)

    if use_genai_balance:
        targets = np.array(ds_tr.targets)
        class_sample_count = np.bincount(targets)
        class_weight = 1.0 / np.maximum(class_sample_count, 1)
        sample_weights = class_weight[targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=pin_mem)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_mem)

    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_mem)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=epochs) if cosine else None

    scaler = torch.amp.GradScaler('cuda', enabled=(amp and is_cuda))
    loss_fn = nn.CrossEntropyLoss()

    # optional linear probe
    if linear_probe_epochs and linear_probe_epochs > 0:
        for p in model.parameters(): p.requires_grad = False
        head = getattr(model, "head", None)
        if head is not None:
            for p in head.parameters(): p.requires_grad = True
        for _ in range(linear_probe_epochs):
            model.train()
            for x, y in dl_tr:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(amp and is_cuda)):
                    logits = model(x); loss = loss_fn(logits, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        for p in model.parameters(): p.requires_grad = True
        opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = CosineAnnealingLR(opt, T_max=epochs) if cosine else None

    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(amp and is_cuda)):
                logits = model(x); loss = loss_fn(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if sched: sched.step()

        y_true, y_pred, p1 = _eval_torch_cls(model, dl_va, device)
        f1  = f1_score(y_true, y_pred)
        improved = f1 > best_f1 + 1e-6
        if improved:
            best_f1 = f1; bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred, p1 = _eval_torch_cls(model, dl_va, device)
    return model, y_true, y_pred, p1

# ----------------------------------------------------------------------------- #
# Data class & utilities
# ----------------------------------------------------------------------------- #
@dataclass
class PaperProfile:
    paper_id: str
    modality: str          # "tabular" | "imaging"
    dataset: Any           # key in datasets.yaml OR inline dict
    model: Dict[str, Any]  # {name, params}
    synth: Dict[str, Any]  # {enabled, ...}
    metrics: Dict[str, Any]# {threshold_tuning, tune:{...}, ci:{...}, seed, auto_doc}

def _boot_ci(y_true, y_pred, metric_fn, B=1000, alpha=0.05, seed=1337):
    rng = np.random.default_rng(seed)
    n = len(y_true); idx = np.arange(n)
    stats = []
    for _ in range(B):
        b = rng.choice(idx, size=n, replace=True)
        stats.append(metric_fn(np.array(y_true)[b], np.array(y_pred)[b]))
    lo, hi = np.quantile(stats, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)

def _maybe_tune_tabular(model_spec: ModelSpec, Xtr, ytr, tune_cfg: Dict[str,Any]):
    if not tune_cfg or not bool(tune_cfg.get("enabled", False)):
        return make_tabular(model_spec), {}
    method = str(tune_cfg.get("method", "grid")).lower()
    cv = int(tune_cfg.get("cv", 5))
    scoring_name = str(tune_cfg.get("scoring", "f1")).lower()
    scoring = make_scorer(f1_score) if scoring_name == "f1" else make_scorer(accuracy_score)
    base = make_tabular(model_spec)
    grid = tune_cfg.get("param_grid") or tune_cfg.get("search_space") or {}

    if not grid:
        if model_spec.name.lower() == "xgboost":
            grid = {"n_estimators":[200,300,500],"max_depth":[3,4,5],
                    "learning_rate":[0.05,0.08,0.1],"subsample":[0.7,0.9,1.0],
                    "colsample_bytree":[0.7,0.9,1.0],"min_child_weight":[1,3,5]}
        elif model_spec.name.lower() in ("random_forest","rf"):
            grid = {"n_estimators":[200,400,600],"max_depth":[None,8,12,16],
                    "min_samples_split":[2,5,10]}
        else:
            grid = {"C":[0.1,0.3,1.0,3.0,10.0]}

    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1337)
    if method == "randomized":
        n_iter = int(tune_cfg.get("n_iter", 20))
        search = RandomizedSearchCV(base, grid, n_iter=n_iter, scoring=scoring, cv=kfold,
                                    refit=True, n_jobs=-1, random_state=1337)
    else:
        search = GridSearchCV(base, grid, scoring=scoring, cv=kfold, refit=True, n_jobs=-1)
    search.fit(Xtr, ytr)
    best_est = search.best_estimator_
    info = {"tune_enabled": True, "method": method, "cv": cv,
            "scoring": scoring_name, "best_params": search.best_params_,
            "best_score_cv": float(search.best_score_)}
    return best_est, info

def _maybe_explain_tabular(model, Xtr, Xte, yte, run_dir: Path, xai_cfg: Dict[str, Any]):
    xai_cfg = xai_cfg or {}
    if not bool(xai_cfg.get("enabled", False)):
        return {"enabled": False}

    xdir = run_dir / "xai"
    xdir.mkdir(parents=True, exist_ok=True)
    method = str(xai_cfg.get("method", "shap")).lower()
    top_k = int(xai_cfg.get("top_k", 20))
    summary = {"enabled": True, "method": method, "artifacts": {}}

    def _save_top(df_imp: pd.DataFrame, fname: str):
        top = df_imp.sort_values("importance", ascending=False).head(top_k)
        out = xdir / fname
        top.to_csv(out, index=False)
        summary["artifacts"]["top_features_csv"] = str(out)
        summary["top_features"] = top.to_dict(orient="records")

    used_perm_fallback = False
    if method == "shap":
        try:
            import shap
            explainer = None
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(Xte)
            except Exception:
                try:
                    explainer = shap.LinearExplainer(model, Xtr)
                    shap_values = explainer.shap_values(Xte)
                except Exception:
                    bsz = int(xai_cfg.get("sample_background", 200))
                    bg = Xtr.sample(min(bsz, len(Xtr)), random_state=1337)
                    explainer = shap.KernelExplainer(
                        model.predict_proba if hasattr(model, "predict_proba") else model.predict, bg
                    )
                    xts = Xte.sample(min(200, len(Xte)), random_state=1337)
                    shap_values = explainer.shap_values(xts)
                    Xte = xts

            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values

            imp = pd.DataFrame({"feature": Xte.columns, "importance": np.abs(sv).mean(axis=0)})
            _save_top(imp, "shap_feature_importance.csv")

            try:
                plt.figure()
                shap.summary_plot(sv, Xte, show=False)
                outp = xdir / "shap_summary.png"
                plt.tight_layout(); plt.savefig(outp, dpi=200); plt.close()
                summary["artifacts"]["summary_png"] = str(outp)
            except Exception:
                pass

            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(Xte)[:, 1]; idx = int(np.argmax(proba))
                else:
                    idx = 0
                plt.figure()
                shap.plots._waterfall.waterfall_legacy(
                    explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                    sv[idx], feature_names=list(Xte.columns), show=False
                )
                outw = xdir / "shap_waterfall.png"
                plt.tight_layout(); plt.savefig(outw, dpi=200); plt.close()
                summary["artifacts"]["waterfall_png"] = str(outw)
            except Exception:
                pass

            summary["engine"] = "shap"
        except Exception as e:
            print(f"[XAI] SHAP failed ({e}); falling back to permutation importance")
            used_perm_fallback = True

    if method == "perm" or used_perm_fallback:
        r = permutation_importance(model, Xte, yte, n_repeats=10, random_state=1337, scoring="f1")
        imp = pd.DataFrame({"feature": Xte.columns, "importance": r.importances_mean})
        _save_top(imp, "perm_feature_importance.csv")
        summary["engine"] = "permutation_importance"

    return summary

# ----------------------------------------------------------------------------- #
# Main entry
# ----------------------------------------------------------------------------- #
def run_from_profile(profile_yaml: str, return_results=False):
    prof_dict = _yml(CFG_DIR / "papers" / profile_yaml)
    prof = PaperProfile(
        paper_id=prof_dict["paper_id"],
        modality=prof_dict["modality"],
        dataset=prof_dict["dataset"],
        model=prof_dict["model"],
        synth=prof_dict.get("synth", {}) or {},
        metrics=prof_dict.get("metrics", {}) or {}
    )
    reg = _reg()

    # Allow inline dataset dict in profile
    if isinstance(prof.dataset, dict):
        ds_dict = prof.dataset
        ds_name = ds_dict.get("name", "inline_dataset")
        reg[ds_name] = {
            "path": ds_dict["path"],
            "label_col": ds_dict.get("label_col", "label"),
            "positive_value": ds_dict.get("positive_value", 1),
        }
        prof.dataset = ds_name

    if prof.dataset not in reg:
        _fail(f"Dataset '{prof.dataset}' not found in datasets.yaml")
    seed = int(prof.metrics.get("seed", 1337)); _seed(seed)

    tune_info = {}
    synth_applied = False
    train_counts_before, train_counts_after = {}, {}

    # ---------------------- TABULAR ---------------------- #
    if prof.modality == "tabular":
        entry = reg[prof.dataset]
        csv_path = resolve_path(entry["path"], expect_file=True)
        df = pd.read_csv(csv_path)

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        label_col = (entry.get("label_col") or "").strip().lower().replace(" ", "_")
        if not label_col:
            candidates = ["label","status","target","class","outcome","y"]
            label_col = next((c for c in candidates if c in df.columns), None)
        if not label_col:
            _fail(f"Could not locate a label column in {csv_path}. Available: {list(df.columns)}")

        id_like = [c for c in ["id","patient_id","sample_id","record_id"] if c in df.columns]

        X = df.drop(columns=id_like + [label_col])
        y = df[label_col]

        if y.dtype == "O":
            y = y.astype(str).str.strip().str.lower().map({
                "1":1, "0":0, "true":1, "false":0, "yes":1, "no":0,
                "positive":1, "negative":0, "mpox":1, "non_mpox":0, "pos":1, "neg":0
            })
        if y.isna().any():
            bad = sorted(y[y.isna()].index.tolist()[:5])
            raise ValueError(
                f"Label column '{label_col}' contains unmapped values. "
                f"Examples rows: {bad}. Unique values: {sorted(df[label_col].astype(str).unique().tolist())}"
            )
        y = y.astype(int)

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

        train_counts_before = _py_counts(dict(zip(*np.unique(ytr, return_counts=True))))
        if prof.synth.get("enabled", False):
            mult = float(prof.synth.get("minority_multiplier", 2.0))
            balance_to_max = bool(prof.synth.get("balance_to_max", False))
            target_ratio   = prof.synth.get("target_ratio", None)
            target_ratio   = float(target_ratio) if target_ratio is not None else None
            Xtr, ytr = smote_or_oversample(
                Xtr, ytr, multiplier=mult, seed=seed,
                balance_to_max=balance_to_max, target_ratio=target_ratio
            )
        train_counts_after  = _py_counts(dict(zip(*np.unique(ytr, return_counts=True))))
        synth_applied = bool(prof.synth.get("enabled", False)) and (train_counts_before != train_counts_after)
        print(f"[DOC] synth_enabled={prof.synth.get('enabled', False)}, "
              f"balance_to_max={prof.synth.get('balance_to_max', False)}, "
              f"target_ratio={prof.synth.get('target_ratio', None)} | "
              f"train_counts_before={train_counts_before} -> after={train_counts_after}")

        model_spec = ModelSpec(**prof.model)
        model, tune_info = _maybe_tune_tabular(model_spec, Xtr, ytr, prof.metrics.get("tune", {}))
        model.fit(Xtr, ytr)

        proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None
        yhat  = (proba >= 0.5).astype(int) if proba is not None else model.predict(Xte)

    # ---------------------- IMAGING ---------------------- #
    elif prof.modality == "imaging":
        entry = reg[prof.dataset]
        mk = make_imaging(ModelSpec(**prof.model))
        if mk is None:
            _fail(f"Unknown imaging model spec: {prof.model}")

        if isinstance(mk, dict) and mk.get("engine") == "torch_vit":
            vit        = mk["model"]
            preprocess = mk["preprocess"]
            hparams    = mk.get("params", {})
            ds_tr, ds_va, tr_counts_b, va_counts = _build_torch_image_datasets(entry, preprocess, seed=seed)

            synth_enabled  = bool(prof.synth.get("enabled", False))
            balance_to_max = bool(prof.synth.get("balance_to_max", False))
            target_ratio   = prof.synth.get("target_ratio", None)
            use_sampler    = synth_enabled and (balance_to_max or target_ratio is not None)

            model, yte, yhat, proba = _train_vit(
                vit, ds_tr, ds_va,
                epochs=int(hparams.get("epochs", 30)),
                batch_size=int(hparams.get("batch_size", 16)),
                lr=float(hparams.get("lr", 3e-4)),
                weight_decay=float(hparams.get("weight_decay", 0.05)),
                cosine=bool(hparams.get("cosine_lr", True)),
                patience=int(hparams.get("patience", 7)),
                amp=bool(hparams.get("amp", True)),
                seed=seed,
                use_genai_balance=use_sampler
            )
            train_counts_before = _py_counts(tr_counts_b)
            train_counts_after  = train_counts_before
            synth_applied = use_sampler
        else:
            root = resolve_path(entry["root"], expect_dir=True)
            ds_tr = ImageFolderDataset(root, "train",
                                       pos_cls=entry["classes"]["positive"], neg_cls=entry["classes"]["negative"],
                                       seed=seed, img_size=int((entry.get("image_size") or 128)))
            ds_va = ImageFolderDataset(root, "val",
                                       pos_cls=entry["classes"]["positive"], neg_cls=entry["classes"]["negative"],
                                       seed=seed, img_size=int((entry.get("image_size") or 128)))

            tr_labels = ds_tr.labels
            tr_lab, tr_cnt = np.unique(tr_labels, return_counts=True)
            train_counts_before = {int(a): int(b) for a, b in zip(tr_lab, tr_cnt)}

            balance_to_max = bool(prof.synth.get("balance_to_max", False))
            target_ratio   = prof.synth.get("target_ratio", None)
            target_ratio   = float(target_ratio) if target_ratio is not None else None
            mult           = float(prof.synth.get("minority_multiplier", 2.0))
            synth_enabled  = bool(prof.synth.get("enabled", False))

            Xtr, ytr = ds_tr.as_features_labels(
                synth_enabled=synth_enabled, synth_multiplier=mult,
                balance_to_max=balance_to_max, target_ratio=target_ratio,
                seed=seed, synth_verbose=True,
            )
            Xte, yte = ds_va.as_features_labels(synth_enabled=False)

            tr_lab2, tr_cnt2 = np.unique(ytr, return_counts=True)
            train_counts_after = {int(a): int(b) for a, b in zip(tr_lab2, tr_cnt2)}
            synth_applied = synth_enabled and (train_counts_before != train_counts_after)
            print(f"[DOC][IMG] synth_enabled={synth_enabled}, balance_to_max={balance_to_max}, "
                  f"target_ratio={target_ratio} | before={train_counts_before} -> after={train_counts_after}")

            model = make_imaging(ModelSpec(**prof.model))
            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None
            yhat  = (proba >= 0.5).astype(int) if proba is not None else model.predict(Xte)
            yte   = yte
    else:
        _fail(f"Unknown modality '{prof.modality}'")

    # ---------------------- Metrics & tuning ---------------------- #
    acc = float(accuracy_score(yte, yhat))
    f1  = float(f1_score(yte, yhat))

    ci  = (prof.metrics.get("ci") or {})
    ci_on = bool(ci.get("enabled", True)); B=int(ci.get("B", 1000)); a=float(ci.get("alpha", 0.05))
    f1_ci = _boot_ci(yte, yhat, f1_score, B=B, alpha=a, seed=seed) if ci_on else None

    tuned = {}
    if bool(prof.metrics.get("threshold_tuning", False)) and proba is not None:
        mode = str(prof.metrics.get("tune_for", "f1")).lower()
        if mode == "accuracy":
            cand = np.linspace(0.05, 0.95, 19)
            accs = [accuracy_score(yte, (proba >= t).astype(int)) for t in cand]
            best_idx = int(np.argmax(accs)); best = float(cand[best_idx])
            yhat_t = (proba >= best).astype(int)
            tuned = {"threshold": best, "acc_tuned": float(accs[best_idx])}
        else:
            prec, rec, thr = precision_recall_curve(yte, proba)
            f1s = 2 * prec * rec / (prec + rec + 1e-12)
            best = float(thr[f1s[:-1].argmax()]) if len(thr) > 0 else 0.5
            yhat_t = (proba >= best).astype(int)
            tuned = {"threshold": best, "f1_tuned": float(f1_score(yte, yhat_t))}

    # ---------------------- Manifest & outputs ---------------------- #
    manif = {
        "run_id": str(int(time.time())),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paper_id": prof.paper_id,
        "modality": prof.modality,
        "dataset_key": prof.dataset,
        "model": prof.model,
        "synth": {**prof.synth, "applied": synth_applied},
        "metrics": {"acc": acc, "f1": f1, "f1_ci": f1_ci, **tuned},
        "data_snapshot": (
            {
                "n_features": int(X.shape[1]),
                "label_col": label_col,
                "train_counts_before": train_counts_before,
                "train_counts_after": train_counts_after,
            } if prof.modality == "tabular" else
            {
                "img_size": int((reg[prof.dataset].get("image_size") or 128)),
                "train_counts_before": train_counts_before,
                "train_counts_after": train_counts_after,
            }
        ),
        "tuning": tune_info or {"tune_enabled": False},
        "config_name": profile_yaml,
    }

    out = RUNS / manif["run_id"]
    out.mkdir(parents=True, exist_ok=True)

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            return super().default(obj)

    (out / "manifest.json").write_text(json.dumps(manif, indent=2, cls=NpEncoder))

    # XAI (tabular)
    if prof.modality == "tabular" and bool(prof.metrics.get("xai", False)):
        try:
            manif["xai"] = _maybe_explain_tabular(model, Xtr, Xte, yte, out, prof.metrics.get("xai_cfg", {}))
            (out / "manifest.json").write_text(json.dumps(manif, indent=2, cls=NpEncoder))
        except Exception as e:
            manif["xai"] = {"enabled": False, "error": str(e)}

    # XAI (imaging) — simple permutation-importance over feature vectors path,
    # or add CAM/LIME in your imaging runners (recommended).
    if prof.modality == "imaging" and bool(prof.metrics.get("xai", False)) and 'Xte' in locals():
        try:
            r = permutation_importance(model, Xte, yte, n_repeats=10, random_state=1337, scoring="f1")
            feat_names = [f"pix{i}" for i in range(Xte.shape[1])]
            imp = pd.DataFrame({"feature": feat_names, "importance": r.importances_mean})
            top_k = int(prof.metrics.get("xai_top_k", 20))
            top = imp.sort_values("importance", ascending=False).head(top_k)
            xai_dir = out / "xai"; xai_dir.mkdir(parents=True, exist_ok=True)
            csv_path = xai_dir / "imaging_perm_feature_importance.csv"; top.to_csv(csv_path, index=False)
            fig_path = xai_dir / "imaging_perm_top.png"
            plt.figure(figsize=(8, 4.5))
            plt.bar(top["feature"][::-1], top["importance"][::-1]); plt.xticks(rotation=45, ha="right")
            plt.tight_layout(); plt.savefig(fig_path, dpi=160); plt.close()
            manif["xai"] = {"enabled": True, "method": "permutation_importance",
                            "top_features": top.to_dict(orient="records"),
                            "csv": str(csv_path), "figure": str(fig_path)}
            (out / "manifest.json").write_text(json.dumps(manif, indent=2, cls=NpEncoder))
        except Exception as e:
            manif["xai"] = {"enabled": False, "error": str(e)}

    if bool(prof.metrics.get("auto_doc", False)):
        try:
            from cbe_repro.reporting.write_docs import write_experiment_cards, write_section_4_2_5_report
            write_experiment_cards(); write_section_4_2_5_report()
        except Exception as e:
            print(f"[Auto-doc skipped] {e}")

    if return_results:
        return manif
    print(json.dumps(manif, indent=2))
