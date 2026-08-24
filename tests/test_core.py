import numpy as np

from baa2.core import AnalysisConfig, analyze_array, auxiva, synthetic_convolutive_problem


def test_auxiva_shapes_and_finiteness():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3, 100)) + 1j * rng.normal(size=(8, 3, 100))
    Y, W, V = auxiva(X, n_iter=2)
    assert Y.shape == (8, 3, 100)
    assert W.shape == (8, 3, 3)
    assert V.shape == (8, 3, 3)
    assert np.isfinite(Y).all()


def test_synthetic_pipeline_produces_three_coordinate_systems():
    data, fs, names, _ = synthetic_convolutive_problem(seed=1, seconds=18)
    cfg = AnalysisConfig(iva_iterations=3, n_states=6, latent_dim=4, max_channels=8)
    out = analyze_array(data, fs, names, cfg)
    assert set(out["analyses"]) == {"PCA-bandpower", "ICA-bandpower", "IVA-broadband"}
    for result in out["analyses"].values():
        assert result["transition_probs"].shape[0] >= 2
        assert len(result["hsl"]) == result["transition_probs"].shape[0]
