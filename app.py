from __future__ import annotations

import gradio as gr
import mne

from baa2.core import AnalysisConfig, EEG_REGIONS, analyze_array, canonical_channel, select_region_channels
from baa2.viz import correlation_figure, sankey_figure, state_map_figure


def load_edf(path: str, region: str, max_seconds: float, cfg: AnalysisConfig):
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.rename_channels({ch: canonical_channel(ch) for ch in raw.ch_names})
    idx = select_region_channels(raw.ch_names, region)
    if not idx:
        raise gr.Error(f"No channels from region {region!r} were found in this EDF.")
    idx = idx[: cfg.max_channels]
    names = [raw.ch_names[i] for i in idx]
    data = raw.get_data(picks=idx)
    fs = float(raw.info["sfreq"])
    if max_seconds and max_seconds > 0:
        n = min(data.shape[1], int(round(max_seconds * fs)))
        data = data[:, :n]
    return data, fs, names


def summary_markdown(result: dict) -> str:
    lines = [
        "## Representation stability receipt",
        "",
        "The H/S/L labels are **descriptive coarse-graining of recurring signal states**. They are not labels for thoughts, consciousness, or anatomical connectivity.",
        "",
        "| representation | frozen split-half transition similarity | occupancy similarity | H | L | S |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, a in result["analyses"].items():
        st = a["stability"]
        m = a["metrics"]
        lines.append(
            f"| {name} | {st['transition_similarity']:.3f} | {st['occupancy_similarity']:.3f} | {m['n_hubs']} | {m['n_loops']} | {m['n_states']} |"
        )
    lines += [
        "",
        f"Channels: **{', '.join(result['channel_names'])}**",
        "",
        "### What the three arms mean",
        "- **PCA-bandpower**: closest control to the old BrainArchitectureAnalyzer: electrode×band power → PCA → states.",
        "- **ICA-bandpower**: independent components of the same power-modulation features → states. This tests whether rotating away higher-order dependence changes the state grammar.",
        "- **IVA-broadband**: complex STFT sensor mixtures → frequency-coupled AuxIVA → source×band trajectories → states. This is the arm that preserves phase/frequency structure and attacks the frequency-permutation problem.",
        "",
        "For the stability scores, the representation, state coordinates, and K-means map are fitted on half A and frozen before half B is replayed.",
        "",
        "The immediate gate is not which plot looks nicer. It is whether the source-space state grammar is more reproducible across time, channel subsets, and later recordings.",
    ]
    return "\n".join(lines)


def analyze_edf(edf_path, region, latent_dim, n_states, iva_iterations, max_channels, max_seconds):
    if not edf_path:
        raise gr.Error("Upload an EDF file first.")
    cfg = AnalysisConfig(
        latent_dim=int(latent_dim),
        n_states=int(n_states),
        iva_iterations=int(iva_iterations),
        max_channels=int(max_channels),
    )
    data, fs, names = load_edf(str(edf_path), region, float(max_seconds), cfg)
    if data.shape[1] < int(fs * 8):
        raise gr.Error("The selected recording segment is too short; use at least about 8 seconds.")
    result = analyze_array(data, fs, names, cfg)

    corr = correlation_figure(
        result["observation_correlation"],
        result["band_feature_names"],
        "Observed electrode×band dependence (not anatomical wiring)",
    )
    pca = result["analyses"]["PCA-bandpower"]
    ica = result["analyses"]["ICA-bandpower"]
    iva = result["analyses"]["IVA-broadband"]
    return (
        summary_markdown(result),
        corr,
        state_map_figure(pca, "PCA-bandpower state space"),
        sankey_figure(pca, "PCA-bandpower H/S/L transition graph"),
        state_map_figure(ica, "ICA-bandpower source-state space"),
        sankey_figure(ica, "ICA-bandpower H/S/L transition graph"),
        state_map_figure(iva, "IVA-broadband source-state space"),
        sankey_figure(iva, "IVA-broadband H/S/L transition graph"),
    )


with gr.Blocks(title="BrainArchitectureAnalyzer2") as app:
    gr.Markdown(
        "# BrainArchitectureAnalyzer2\n"
        "**Observation matrix → demixing transform → latent-source dynamics → H/S/L coarse grammar.**  "
        "This rebuild keeps the old visual idea but makes the coordinate question explicit."
    )
    with gr.Row():
        with gr.Column(scale=1):
            edf = gr.File(label="EEG EDF", file_types=[".edf"], type="filepath")
            region = gr.Dropdown(list(EEG_REGIONS.keys()), value="Occipital", label="Region")
            latent = gr.Slider(2, 16, value=8, step=1, label="State-space dimensions")
            states = gr.Slider(3, 30, value=12, step=1, label="K-means states")
            iva_iter = gr.Slider(5, 80, value=20, step=5, label="AuxIVA iterations")
            max_ch = gr.Slider(2, 24, value=8, step=1, label="Maximum channels")
            max_seconds = gr.Slider(15, 600, value=120, step=15, label="Analyze first N seconds")
            run = gr.Button("Analyze coordinate systems", variant="primary")
        with gr.Column(scale=2):
            summary = gr.Markdown("Upload an EDF to compare PCA, ICA, and IVA state grammars.")

    with gr.Tab("Observed mixtures"):
        corr = gr.Plot(label="Observation dependence matrix")
    with gr.Tab("PCA baseline"):
        with gr.Row():
            pca_map = gr.Plot()
            pca_flow = gr.Plot()
    with gr.Tab("ICA on band-power features"):
        with gr.Row():
            ica_map = gr.Plot()
            ica_flow = gr.Plot()
    with gr.Tab("IVA on complex broadband EEG"):
        with gr.Row():
            iva_map = gr.Plot()
            iva_flow = gr.Plot()

    run.click(
        analyze_edf,
        inputs=[edf, region, latent, states, iva_iter, max_ch, max_seconds],
        outputs=[summary, corr, pca_map, pca_flow, ica_map, ica_flow, iva_map, iva_flow],
    )


if __name__ == "__main__":
    app.launch()
