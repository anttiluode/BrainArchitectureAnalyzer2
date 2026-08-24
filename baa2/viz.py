from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def correlation_figure(C: np.ndarray, names: list[str], title: str = "Observation dependence matrix") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=C, x=names, y=names, zmin=-1, zmax=1, zmid=0, colorscale="RdBu_r"))
    fig.update_layout(title=title, template="plotly_dark", height=620, margin=dict(l=80, r=30, t=70, b=90))
    return fig


def state_map_figure(result: dict, title: str) -> go.Figure:
    z = np.asarray(result["trajectory"])
    labels = np.asarray(result["labels"])
    if z.shape[1] < 2:
        xy = np.c_[z[:, 0], np.zeros(len(z))]
    else:
        xy = z[:, :2]
    token_by_state = {x["state"]: x["token"] for x in result["hsl"]}
    tokens = np.array([token_by_state[int(s)] for s in labels])
    colors = {"H": "#66dd77", "L": "#ff6b6b", "S": "#6ea8fe"}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="lines", line=dict(color="rgba(180,180,180,.22)", width=1), hoverinfo="skip", showlegend=False))
    for tok in ("H", "L", "S"):
        m = tokens == tok
        fig.add_trace(go.Scatter(
            x=xy[m, 0], y=xy[m, 1], mode="markers", name=tok,
            marker=dict(size=7, color=colors[tok]),
            text=[f"frame {i} · state {labels[i]} · {tok}" for i in np.where(m)[0]],
            hovertemplate="%{text}<extra></extra>",
        ))
    stab = result.get("stability", {})
    fig.update_layout(
        title=f"{title}<br><sup>split-half transition similarity={stab.get('transition_similarity', float('nan')):.3f}</sup>",
        template="plotly_dark", height=500, xaxis_title="state coordinate 1", yaxis_title="state coordinate 2",
    )
    return fig


def sankey_figure(result: dict, title: str, threshold_fraction: float = 0.04) -> go.Figure:
    counts = np.asarray(result["transition_counts"], float)
    hsl = result["hsl"]
    labels = [f"{x['token']}{x['state']} ({x['occupancy']})" for x in hsl]
    palette = {"H": "rgba(80,220,110,.85)", "L": "rgba(255,95,95,.85)", "S": "rgba(100,150,255,.8)"}
    node_colors = [palette[x["token"]] for x in hsl]
    mx = counts.max() if counts.size else 0.0
    threshold = mx * threshold_fraction
    src=[]; dst=[]; val=[]; col=[]
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if counts[i, j] > threshold and counts[i, j] > 0:
                src.append(i); dst.append(j); val.append(float(counts[i, j]))
                col.append("rgba(255,130,130,.28)" if i == j else "rgba(135,145,255,.26)")
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, color=node_colors, pad=18, thickness=24, line=dict(color="black", width=.5)),
        link=dict(source=src, target=dst, value=val, color=col),
    ))
    fig.update_layout(title=title, template="plotly_dark", height=620)
    return fig
