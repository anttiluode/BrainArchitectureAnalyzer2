from __future__ import annotations

import json
from pathlib import Path

from baa2.core import AnalysisConfig, analyze_array, synthetic_convolutive_problem


if __name__ == "__main__":
    data, fs, names, _sources = synthetic_convolutive_problem(seed=7, seconds=60)
    cfg = AnalysisConfig(iva_iterations=20, n_states=10, latent_dim=6, max_channels=8)
    result = analyze_array(data, fs, names, cfg)
    receipt = {}
    for name, a in result["analyses"].items():
        receipt[name] = {"metrics": a["metrics"], "stability": a["stability"]}
        print(name, receipt[name])
    Path("synthetic_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
