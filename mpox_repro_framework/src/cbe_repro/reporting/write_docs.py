# -*- coding: utf-8 -*-
from pathlib import Path
import json, yaml, pandas as pd
from datetime import datetime

ROOT = Path.cwd() / "mpox_repro_framework"
SRC  = ROOT / "src"
RUNS = ROOT / "runs"
CFG  = SRC / "cbe_repro" / "configs"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

def _latest_manifests_df():
    rows = []
    for d in RUNS.glob("*"):
        mf = d / "manifest.json"
        if mf.exists():
            row = json.loads(mf.read_text())
            # tolerate old manifests that lack config_name
            if "config_name" not in row:
                row["config_name"] = row.get("paper_id", "unknown")  # or f"{row.get('modality','?')}_unknown.yaml"
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("started").groupby("config_name", as_index=False).tail(1)
    return df.reset_index(drop=True)

def _repro_score(cfg_name: str) -> float:
    try:
        cfg_path = CFG / cfg_name
        reg_path = CFG / "datasets.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        reg = yaml.safe_load(reg_path.read_text())
        bits = []
        bits.append(1.0 if cfg_path.exists() else 0.0)                             # config present
        bits.append(1.0 if isinstance(cfg.get("seed"), int) else 0.0)              # fixed seed
        bits.append(1.0 if cfg.get("dataset") in reg else 0.0)                     # dataset registered
        bits.append(1.0)                                                           # manifest exists (we're reading it)
        synth = (cfg.get("synth", {}) or {}).get("enabled", False)
        bits.append(1.0 if isinstance(synth, bool) else 0.0)                       # augmentation control
        return float(sum(bits)/len(bits))
    except Exception:
        return 0.0

def write_experiment_cards():
    df = _latest_manifests_df()
    if df.empty:
        print("No manifests found."); return
    cards_dir = REPORTS / "cards"
    cards_dir.mkdir(exist_ok=True)
    for _, row in df.iterrows():
        cfg = row["config_name"]
        m = row["metrics"]
        rs = _repro_score(cfg)
        lines = []
        lines.append(f"# Experiment Card — {cfg}")
        lines.append("")
        lines.append(f"- **Started:** {row['started']}")
        lines.append(f"- **ReproScore:** {rs:.2f}")
        lines.append(f"- **Accuracy:** {m['acc']:.3f}")
        lines.append(f"- **F1:** {m['f1']:.3f}")
        lines.append("")
        try:
            ycfg = yaml.safe_load((CFG/cfg).read_text())
            lines.append("## Key Config")
            lines.append(f"- Seed: `{ycfg.get('seed')}`")
            lines.append(f"- Dataset: `{ycfg.get('dataset')}`")
            if "image_size" in ycfg: lines.append(f"- Image size: `{ycfg['image_size']}`")
            synth = ycfg.get("synth", {})
            lines.append(f"- GenAI enabled: `{synth.get('enabled', False)}`")
            if "minority_multiplier" in synth:
                lines.append(f"- GenAI minority_multiplier: `{synth['minority_multiplier']}`")
            model = ycfg.get("model", {})
            lines.append(f"- Model: `{model.get('name')}` params: `{model.get('params')}`")
            # --- XAI summary from manifest ---
            xai = row.get("xai", {}) if isinstance(row.get("xai", {}), dict) else {}
            if xai.get("enabled", False):
                lines.append("## XAI (Explainability)")
                lines.append(f"- Method: `{xai.get('method')}`")
                tf = xai.get("top_features") or []
                if tf:
                    # Show top 10 nicely
                    lines.append("- Top features by mean |SHAP|:")
                    for name, val in tf[:10]:
                        lines.append(f"  - `{name}`: {val:.5f}")
                fig = xai.get("figure")
                if fig and Path(fig).exists():
                    rel = Path(fig).relative_to(ROOT)
                    lines.append("")
                    lines.append(f"![SHAP top features]({rel.as_posix()})")
            else:
                lines.append("## XAI (Explainability)")
                lines.append("_Disabled or unavailable for this run._")
        except Exception:
            pass
        out = cards_dir / f"{cfg.replace('.yaml','')}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print("Wrote card:", out)

def write_section_4_2_5_report():
    df = _latest_manifests_df()
    if df.empty:
        print("No manifests found."); return

    def pick(name): 
        sub = df[df["config_name"]==name]
        return None if sub.empty else sub.iloc[0]

    rows = []
    for stem in ["imaging","tabular"]:
        base = pick(f"{stem}_baseline.yaml")
        gen  = pick(f"{stem}_genai.yaml")
        if base is None or gen is None: continue
        b, g = base["metrics"], gen["metrics"]
        row = {
            "task": stem,
            "baseline_acc": float(b["acc"]), "genai_acc": float(g["acc"]),
            "delta_acc":  float(g["acc"])-float(b["acc"]),
            "baseline_f1": float(b["f1"]),   "genai_f1": float(g["f1"]),
            "delta_f1":   float(g["f1"])-float(b["f1"]),
            "baseline_rs": _repro_score(base["config_name"]),
            "genai_rs":    _repro_score(gen["config_name"]),
            "delta_rs":    _repro_score(gen["config_name"]) - _repro_score(base["config_name"]),
        }
        rows.append(row)
    comp = pd.DataFrame(rows)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("# 4.2.5 Generative AI–Based Interventions for Reproducibility")
    lines.append("")
    lines.append(f"_Auto-generated: {ts}_")
    lines.append("")
    lines.append("## Setup (Reproducibility)")
    lines.append("- Fixed RNG seed (`seed=1337`) in configs.")
    lines.append("- Datasets registered in a central `datasets.yaml`.")
    lines.append("- Each run logs a JSON manifest under `runs/<id>/manifest.json`.")
    lines.append("- **GenAI interventions:** imaging = deterministic oversampling of positives; tabular = SMOTE/oversample with seed.")
    lines.append("")
    lines.append("## Latest Per-Config Metrics")
    lines.append(df[["config_name","metrics","started"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Baseline vs GenAI (Δ)")
    if not comp.empty:
        lines.append(comp.to_markdown(index=False))
        for _, r in comp.iterrows():
            lines.append(f"- **{r['task'].capitalize()}**: F1 {r['baseline_f1']:.3f} → {r['genai_f1']:.3f} (Δ {r['delta_f1']:+.3f}); "
                         f"Acc {r['baseline_acc']:.3f} → {r['genai_acc']:.3f} (Δ {r['delta_acc']:+.3f}); "
                         f"ReproScore {r['baseline_rs']:.2f} → {r['genai_rs']:.2f} (Δ {r['delta_rs']:+.2f}).")
    else:
        lines.append("_Not enough paired runs to compute deltas._")
    lines.append("")
    lines.append("## Transparency (Auto-Docs)")
    lines.append("- Per-experiment cards with key config knobs are in `reports/cards/`.")

    lines.append("")
    lines.append("## XAI Highlights")
    for stem in ["imaging","tabular"]:
        base = pick(f"{stem}_baseline.yaml")
        gen  = pick(f"{stem}_genai.yaml")
        if base is None or gen is None:
            continue
        bx = base.get("xai", {}) if isinstance(base.get("xai", {}), dict) else {}
        gx = gen.get("xai", {}) if isinstance(gen.get("xai", {}), dict) else {}
        if bx.get("enabled") or gx.get("enabled"):
            lines.append(f"### {stem.capitalize()}")
            if bx.get("enabled"):
                btop = ", ".join([f"{n}({v:.3f})" for n,v in (bx.get('top_features') or [])[:5]])
                lines.append(f"- Baseline top features: {btop if btop else '_n/a_'}")
            else:
                lines.append("- Baseline XAI: _disabled_")
            if gx.get("enabled"):
                gtop = ", ".join([f"{n}({v:.3f})" for n,v in (gx.get('top_features') or [])[:5]])
                lines.append(f"- GenAI top features: {gtop if gtop else '_n/a_'}")
            else:
                lines.append("- GenAI XAI: _disabled_")

    out = REPORTS / "section_4_2_5.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote section report:", out)
