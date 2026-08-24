from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from scipy import signal, stats
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler

EPS = 1e-10

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

EEG_REGIONS = {
    "All": [],
    "Occipital": ["O1", "O2", "OZ", "POZ", "PO3", "PO4", "PO7", "PO8"],
    "Temporal": ["T7", "T8", "TP7", "TP8", "FT7", "FT8"],
    "Parietal": ["P1", "P2", "P3", "P4", "PZ", "CP1", "CP2"],
    "Frontal": ["FP1", "FP2", "FZ", "F1", "F2", "F3", "F4"],
    "Central": ["C1", "C2", "C3", "C4", "CZ", "FC1", "FC2"],
}


@dataclass
class AnalysisConfig:
    fs: float = 128.0
    l_freq: float = 1.0
    h_freq: float = 45.0
    stft_window_s: float = 1.0
    stft_hop_s: float = 0.25
    latent_dim: int = 8
    n_states: int = 12
    iva_iterations: int = 35
    random_state: int = 42
    max_channels: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_channel(name: str) -> str:
    return name.strip().replace(".", "").replace(" ", "").upper()


def select_region_channels(channel_names: list[str], region: str) -> list[int]:
    canon = [canonical_channel(x) for x in channel_names]
    if region == "All" or region not in EEG_REGIONS:
        return list(range(len(channel_names)))
    wanted = set(EEG_REGIONS[region])
    return [i for i, ch in enumerate(canon) if ch in wanted]


def prepare_array(data: np.ndarray, fs: float, cfg: AnalysisConfig) -> tuple[np.ndarray, float]:
    """Resample and band-limit a channels x samples array."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("data must have shape channels x samples")
    if x.shape[0] < 2:
        raise ValueError("at least two channels are required")
    if not np.isfinite(x).all():
        x = np.nan_to_num(x)

    target_fs = float(cfg.fs)
    if abs(float(fs) - target_fs) > 1e-9:
        from fractions import Fraction

        ratio = Fraction(target_fs / float(fs)).limit_denominator(1000)
        x = signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=1)
        fs = target_fs

    ny = 0.5 * float(fs)
    lo = max(0.01, cfg.l_freq) / ny
    hi = min(cfg.h_freq, ny * 0.98) / ny
    if not 0 < lo < hi < 1:
        raise ValueError("invalid band-pass settings for sampling rate")
    sos = signal.butter(4, [lo, hi], btype="bandpass", output="sos")
    x = signal.sosfiltfilt(sos, x, axis=1)
    x -= np.mean(x, axis=1, keepdims=True)
    scale = np.std(x, axis=1, keepdims=True)
    x /= np.where(scale > EPS, scale, 1.0)
    return x, float(fs)


def complex_stft(data: np.ndarray, fs: float, cfg: AnalysisConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X[f, channel, time], frequency bins, and frame times."""
    nperseg = max(16, int(round(cfg.stft_window_s * fs)))
    hop = max(1, int(round(cfg.stft_hop_s * fs)))
    noverlap = max(0, nperseg - hop)
    f, t, z = signal.stft(
        data,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
        axis=-1,
    )
    keep = (f >= cfg.l_freq) & (f <= min(cfg.h_freq, fs / 2.0))
    X = np.transpose(z[:, keep, :], (1, 0, 2)).astype(np.complex128, copy=False)
    return X, f[keep], t


def bandpower_features(X: np.ndarray, freqs: np.ndarray, channel_names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Log band powers from a complex STFT. Output is time x (channel*band)."""
    _, C, T = X.shape
    power = np.abs(X) ** 2
    feats: list[np.ndarray] = []
    names: list[str] = []
    channel_names = channel_names or [f"CH{i}" for i in range(C)]
    for c in range(C):
        for band, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi if band != "gamma" else freqs <= hi)
            if not np.any(mask):
                v = np.zeros(T)
            else:
                v = np.log1p(np.mean(power[mask, c, :], axis=0))
            feats.append(v)
            names.append(f"{channel_names[c]}-{band}")
    return np.stack(feats, axis=1), names


def correlation_fingerprint(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, float)
    if x.shape[0] < 2:
        return np.eye(x.shape[1])
    C = np.corrcoef(x, rowvar=False)
    return np.nan_to_num(C)


def whiten_frequency_bins(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric whitening independently for each frequency bin."""
    F, C, T = X.shape
    Xw = np.empty_like(X, dtype=np.complex128)
    V = np.empty((F, C, C), dtype=np.complex128)
    for f in range(F):
        cov = (X[f] @ X[f].conj().T) / max(1, T)
        cov += EPS * np.eye(C)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals.real, EPS)
        vf = (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.conj().T
        V[f] = vf
        Xw[f] = vf @ X[f]
    return Xw, V


def auxiva(X: np.ndarray, n_iter: int = 35) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Determined complex AuxIVA with iterative projection.

    Parameters
    ----------
    X : ndarray, shape (frequency, sensor, frame)
        Complex STFT sensor mixtures.

    Returns
    -------
    Y : frequency x source x frame
    W : frequency x source x sensor (acts on whitened X)
    V : frequency x sensor x sensor whitening matrices

    Notes
    -----
    This is intentionally a compact research implementation, not a production
    speech separator. Components are unordered and have arbitrary scale/phase.
    """
    Xw, Vwhite = whiten_frequency_bins(np.asarray(X, np.complex128))
    F, C, T = Xw.shape
    W = np.tile(np.eye(C, dtype=np.complex128), (F, 1, 1))
    eye = np.eye(C, dtype=np.complex128)

    for _ in range(int(n_iter)):
        Y = np.einsum("fsc,fct->fst", W, Xw, optimize=True)
        r = np.sqrt(np.sum(np.abs(Y) ** 2, axis=0) + EPS)
        phi = 1.0 / np.maximum(r, 1e-7)

        for s in range(C):
            for f in range(F):
                Xf = Xw[f]
                weighted = Xf * phi[s][None, :]
                cov = (weighted @ Xf.conj().T) / max(1, T)
                cov += EPS * eye
                A = W[f] @ cov
                try:
                    w = np.linalg.solve(A, eye[:, s])
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(A, eye[:, s], rcond=None)[0]
                denom = np.sqrt(np.real(w.conj().T @ cov @ w) + EPS)
                w /= denom
                W[f, s, :] = w.conj()

    Y = np.einsum("fsc,fct->fst", W, Xw, optimize=True)
    return Y, W, Vwhite


def iva_source_band_features(Y: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Convert broadband IVA sources into source x band energy trajectories."""
    _, S, T = Y.shape
    power = np.abs(Y) ** 2
    cols: list[np.ndarray] = []
    names: list[str] = []
    for s in range(S):
        for band, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi if band != "gamma" else freqs <= hi)
            if np.any(mask):
                v = np.log1p(np.mean(power[mask, s, :], axis=0))
            else:
                v = np.zeros(T)
            cols.append(v)
            names.append(f"IC{s+1}-{band}")
    return np.stack(cols, axis=1), names


def pca_representation(features: np.ndarray, latent_dim: int, random_state: int = 42) -> tuple[np.ndarray, dict[str, Any]]:
    x = StandardScaler().fit_transform(features)
    n = max(2, min(int(latent_dim), x.shape[0] - 1, x.shape[1]))
    pca = PCA(n_components=n, random_state=random_state)
    z = pca.fit_transform(x)
    return z, {"explained_variance": pca.explained_variance_ratio_.tolist()}


def ica_representation(features: np.ndarray, latent_dim: int, random_state: int = 42) -> tuple[np.ndarray, dict[str, Any]]:
    x = StandardScaler().fit_transform(features)
    n = max(2, min(int(latent_dim), x.shape[0] - 1, x.shape[1]))
    ica = FastICA(
        n_components=n,
        whiten="unit-variance",
        random_state=random_state,
        max_iter=1500,
        tol=1e-4,
    )
    z = ica.fit_transform(x)
    return z, {"n_iter": int(ica.n_iter_), "mixing_shape": list(ica.mixing_.shape)}


def compress_for_states(features: np.ndarray, latent_dim: int, random_state: int = 42) -> np.ndarray:
    x = StandardScaler().fit_transform(features)
    n = max(2, min(int(latent_dim), x.shape[0] - 1, x.shape[1]))
    if x.shape[1] <= n:
        return x
    return PCA(n_components=n, random_state=random_state).fit_transform(x)


def transition_counts(labels: np.ndarray, n_states: int) -> np.ndarray:
    T = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(labels[:-1], labels[1:]):
        T[int(a), int(b)] += 1.0
    return T


def row_normalize(T: np.ndarray) -> np.ndarray:
    sums = T.sum(axis=1, keepdims=True)
    return np.divide(T, sums, out=np.zeros_like(T, dtype=float), where=sums > 0)


def classify_hsl(labels: np.ndarray, counts: np.ndarray) -> list[dict[str, Any]]:
    probs = row_normalize(counts)
    n = counts.shape[0]
    occupancy = np.bincount(labels, minlength=n).astype(float)
    incoming = counts.sum(axis=0)
    outgoing = counts.sum(axis=1)
    connectivity = incoming + outgoing
    self_p = np.diag(probs)

    hub_threshold = float(np.percentile(connectivity, 75)) if n else 0.0
    loop_threshold = max(0.25, float(np.percentile(self_p, 75)) if n else 0.25)
    out: list[dict[str, Any]] = []
    for i in range(n):
        if self_p[i] >= loop_threshold and self_p[i] > 0:
            token = "L"
        elif connectivity[i] >= hub_threshold and connectivity[i] > 0:
            token = "H"
        else:
            token = "S"
        out.append(
            {
                "state": i,
                "token": token,
                "occupancy": int(occupancy[i]),
                "self_probability": float(self_p[i]),
                "incoming": float(incoming[i]),
                "outgoing": float(outgoing[i]),
                "connectivity": float(connectivity[i]),
            }
        )
    return out


def state_metrics(labels: np.ndarray, counts: np.ndarray, hsl: list[dict[str, Any]]) -> dict[str, Any]:
    n = counts.shape[0]
    occ = np.bincount(labels, minlength=n).astype(float)
    occ /= max(1.0, occ.sum())
    nz = occ[occ > 0]
    probs = row_normalize(counts)
    pflat = probs[probs > 0]
    return {
        "state_entropy": float(stats.entropy(nz)) if len(nz) else 0.0,
        "transition_entropy": float(stats.entropy(pflat)) if len(pflat) else 0.0,
        "n_hubs": int(sum(x["token"] == "H" for x in hsl)),
        "n_loops": int(sum(x["token"] == "L" for x in hsl)),
        "n_states": int(sum(x["token"] == "S" for x in hsl)),
    }


def fit_states(features: np.ndarray, n_states: int, latent_dim: int, random_state: int = 42) -> dict[str, Any]:
    z = compress_for_states(features, latent_dim, random_state)
    k = max(2, min(int(n_states), max(2, len(z) // 8)))
    km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = km.fit_predict(z)
    counts = transition_counts(labels, k)
    probs = row_normalize(counts)
    hsl = classify_hsl(labels, counts)
    metrics = state_metrics(labels, counts, hsl)
    return {
        "trajectory": z,
        "labels": labels,
        "centers": km.cluster_centers_,
        "transition_counts": counts,
        "transition_probs": probs,
        "hsl": hsl,
        "metrics": metrics,
    }


def transition_similarity(A: np.ndarray, B: np.ndarray) -> float:
    a = row_normalize(A).ravel()
    b = row_normalize(B).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > EPS else 0.0


def split_half_state_stability(features: np.ndarray, n_states: int, latent_dim: int, random_state: int = 42) -> dict[str, float]:
    """Fit coordinates+states on first half, replay the same map on second half."""
    x = np.asarray(features, float)
    mid = len(x) // 2
    if mid < max(20, n_states * 2) or len(x) - mid < 10:
        return {"transition_similarity": float("nan"), "occupancy_similarity": float("nan")}

    scaler = StandardScaler().fit(x[:mid])
    a = scaler.transform(x[:mid])
    b = scaler.transform(x[mid:])
    n = max(2, min(int(latent_dim), a.shape[0] - 1, a.shape[1]))
    pca = PCA(n_components=n, random_state=random_state).fit(a)
    za, zb = pca.transform(a), pca.transform(b)
    k = max(2, min(int(n_states), max(2, len(za) // 8)))
    km = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit(za)
    la, lb = km.labels_, km.predict(zb)
    Ta, Tb = transition_counts(la, k), transition_counts(lb, k)

    oa = np.bincount(la, minlength=k).astype(float) + EPS
    ob = np.bincount(lb, minlength=k).astype(float) + EPS
    oa /= oa.sum()
    ob /= ob.sum()
    js = float(jensenshannon(oa, ob, base=2.0))
    return {
        "transition_similarity": transition_similarity(Ta, Tb),
        "occupancy_similarity": float(1.0 - js),
    }


def analyze_array(
    data: np.ndarray,
    fs: float,
    channel_names: list[str] | None = None,
    cfg: AnalysisConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or AnalysisConfig()
    x, fs = prepare_array(data, fs, cfg)
    if x.shape[0] > cfg.max_channels:
        x = x[: cfg.max_channels]
        if channel_names:
            channel_names = channel_names[: cfg.max_channels]
    channel_names = channel_names or [f"CH{i+1}" for i in range(x.shape[0])]

    X, freqs, times = complex_stft(x, fs, cfg)
    band, band_names = bandpower_features(X, freqs, channel_names)
    corr = correlation_fingerprint(band)

    pca_z, pca_meta = pca_representation(band, cfg.latent_dim, cfg.random_state)
    ica_z, ica_meta = ica_representation(band, cfg.latent_dim, cfg.random_state)
    Y, W, Vwhite = auxiva(X, cfg.iva_iterations)
    iva_band, iva_names = iva_source_band_features(Y, freqs)

    representations = {
        "PCA-bandpower": pca_z,
        "ICA-bandpower": ica_z,
        "IVA-broadband": iva_band,
    }
    rep_meta = {
        "PCA-bandpower": pca_meta,
        "ICA-bandpower": ica_meta,
        "IVA-broadband": {
            "n_sources": int(Y.shape[1]),
            "n_frequency_bins": int(Y.shape[0]),
            "demixer_shape": list(W.shape),
        },
    }

    analyses: dict[str, Any] = {}
    for name, feats in representations.items():
        st = fit_states(feats, cfg.n_states, cfg.latent_dim, cfg.random_state)
        stability = split_half_state_stability(feats, cfg.n_states, cfg.latent_dim, cfg.random_state)
        st["stability"] = stability
        st["representation_meta"] = rep_meta[name]
        analyses[name] = st

    return {
        "config": cfg.to_dict(),
        "fs": fs,
        "channel_names": channel_names,
        "times": times,
        "freqs": freqs,
        "band_feature_names": band_names,
        "iva_feature_names": iva_names,
        "observation_correlation": corr,
        "analyses": analyses,
        "iva": {"W": W, "whitener": Vwhite, "Y": Y},
    }


def synthetic_convolutive_problem(seed: int = 0, fs: float = 128.0, seconds: float = 90.0) -> tuple[np.ndarray, float, list[str], np.ndarray]:
    """Two-source, four-sensor synthetic EEG-ish convolutive mixture for smoke tests."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs

    env1 = signal.savgol_filter(rng.standard_normal(n), 129, 3)
    env2 = signal.savgol_filter(rng.standard_normal(n), 129, 3)
    env1 = np.tanh(env1 * 1.7)
    env2 = np.tanh(env2 * 1.7)
    s1 = env1 * (np.sin(2 * np.pi * 9.5 * t) + 0.45 * np.sin(2 * np.pi * 19.0 * t + 0.2))
    s2 = env2 * (np.sin(2 * np.pi * 6.0 * t + 0.5) + 0.55 * np.sin(2 * np.pi * 27.0 * t))
    s1 += 0.12 * rng.standard_normal(n)
    s2 += 0.12 * rng.standard_normal(n)
    S = np.stack([s1, s2])

    delays = [[0, 5], [3, 0], [8, 2], [1, 11]]
    gains = [[1.0, 0.75], [0.60, 1.0], [0.85, -0.55], [-0.45, 0.90]]
    X = np.zeros((4, n), float)
    for c in range(4):
        for j in range(2):
            d = delays[c][j]
            delayed = np.concatenate([np.zeros(d), S[j, : n - d]]) if d else S[j]
            X[c] += gains[c][j] * delayed
        X[c] += 0.04 * rng.standard_normal(n)
    return X, fs, ["F3", "F4", "O1", "O2"], S
