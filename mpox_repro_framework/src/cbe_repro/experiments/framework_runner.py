# -*- coding: utf-8 -*-
import json, datetime
from pathlib import Path
from . import run_imaging, run_tabular

def run_all():
    results = {}

    # Imaging baseline
    res_img_base = run_imaging.main("imaging_baseline.yaml", return_results=True)
    results["imaging_baseline"] = res_img_base

    # Imaging + GenAI augmentation
    res_img_gen = run_imaging.main("imaging_genai.yaml", return_results=True)
    results["imaging_genai"] = res_img_gen

    # Tabular baseline
    res_tab_base = run_tabular.main("tabular_baseline.yaml", return_results=True)
    results["tabular_baseline"] = res_tab_base

    # Tabular + GenAI augmentation
    res_tab_gen = run_tabular.main("tabular_genai.yaml", return_results=True)
    results["tabular_genai"] = res_tab_gen

    # Save combined results
    out_path = Path.cwd() / "mpox_repro_framework" / "reports" / "results_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "timestamp": str(datetime.datetime.now()),
        "results": results
    }, indent=2))
    print(f"✅ Saved combined results to {out_path}")
    return results

if __name__ == "__main__":
    run_all()
