from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne

from baa2.core import AnalysisConfig, analyze_array, canonical_channel, select_region_channels


def compact_receipt(result: dict) -> dict:
    out = {
        "config": result["config"],
        "fs": result["fs"],
        "channels": result["channel_names"],
        "representations": {},
    }
    for name, a in result["analyses"].items():
        out["representations"][name] = {
            "metrics": a["metrics"],
            "stability": a["stability"],
            "hsl": a["hsl"],
            "transition_probs": a["transition_probs"].tolist(),
            "representation_meta": a["representation_meta"],
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="BrainArchitectureAnalyzer2 CLI")
    ap.add_argument("edf")
    ap.add_argument("--region", default="All")
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--states", type=int, default=12)
    ap.add_argument("--latent", type=int, default=8)
    ap.add_argument("--iva-iterations", type=int, default=35)
    ap.add_argument("--max-channels", type=int, default=12)
    ap.add_argument("--out", default="baa2_receipt.json")
    args = ap.parse_args()

    cfg = AnalysisConfig(
        n_states=args.states,
        latent_dim=args.latent,
        iva_iterations=args.iva_iterations,
        max_channels=args.max_channels,
    )
    raw = mne.io.read_raw_edf(args.edf, preload=True, verbose="ERROR")
    raw.rename_channels({ch: canonical_channel(ch) for ch in raw.ch_names})
    idx = select_region_channels(raw.ch_names, args.region)
    if not idx:
        raise SystemExit(f"No channels found for region {args.region}")
    idx = idx[: cfg.max_channels]
    fs = float(raw.info["sfreq"])
    data = raw.get_data(picks=idx)
    if args.seconds > 0:
        data = data[:, : min(data.shape[1], int(args.seconds * fs))]
    names = [raw.ch_names[i] for i in idx]
    result = analyze_array(data, fs, names, cfg)
    receipt = compact_receipt(result)
    Path(args.out).write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print("BrainArchitectureAnalyzer2")
    for name, a in receipt["representations"].items():
        s = a["stability"]
        m = a["metrics"]
        print(
            f"{name:18s} transition={s['transition_similarity']:.3f} "
            f"occupancy={s['occupancy_similarity']:.3f} H/L/S={m['n_hubs']}/{m['n_loops']}/{m['n_states']}"
        )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
