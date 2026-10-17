"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import prw_constants as C
from prw_evaluation import Analysis, de, lam_text, noise_label, sci

LOG_FLOOR = 1e-17


def _ref_line(fig, value, text, position, axis, log=True, **line_kwargs):
    """Referenzlinie mit Beschriftung auf einer logarithmischen Achse: die Linie liegt auf dem Datenwert, die Beschriftung braucht log10 des Werts
    (add_hline und add_vline setzen beide auf denselben Wert, die Beschriftung sitzt dann falsch und dehnt die Achse); log=False für lineare Achsen."""
    if log and not value > 0.0:
        return
    pos = math.log10(value) if log else value
    top, left = "top" in position, "left" in position
    if axis == "y":
        fig.add_hline(y=value, **line_kwargs)
        if text:
            fig.add_annotation(xref="paper", yref="y", x=0 if left else 1, y=pos, text=text, showarrow=False, xanchor="left" if left else "right", yanchor="bottom" if top else "top")
    else:
        fig.add_vline(x=value, **line_kwargs)
        if text:
            fig.add_annotation(xref="x", yref="paper", x=pos, y=1 if top else 0, text=text, showarrow=False, xanchor="right" if left else "left", yanchor="top" if top else "bottom")


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, a.settings.n)
    return pd.DataFrame({
        "Punkt": [f"{i + 1}" for i in range(k)],
        "Ort s": a.s[:k],
        "Wahrheit x": a.x[:k],
        "Messwert b": a.b[:k],
        "Gewählte Regel": a.x_rule[:k],
        "Orakel (Realisierung)": a.x_oracle[:k],
    })


def style_table(df: pd.DataFrame):
    return df.style.format({"Ort s": lambda v: de(v, 3), "Wahrheit x": lambda v: de(v, 3), "Messwert b": lambda v: de(v, 4), "Gewählte Regel": lambda v: de(v, 3),
                            "Orakel (Realisierung)": lambda v: de(v, 3)})


def _vline(fig: go.Figure, x: float, text: str, color: str, position: str = "top left", dash: str = "solid") -> None:
    fig.add_vline(x=x, line=dict(color=color, width=2, dash=dash), annotation_text=text, annotation_position=position)


def _lam_axis(fig: go.Figure) -> None:
    fig.update_xaxes(title="Strafparameter λ", type="log", tickformat=".0e")


def build_recon_chart(a: Analysis) -> go.Figure:
    """Wahrheit, gewählte Regel, Realisierungsorakel und (blass) die direkte Inversion; die Achse ist fest."""
    lo, hi = C.SHOW_RANGE
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.x_naive, mode="lines", name="direkte Inversion", line=dict(color=C.COLOR_NAIVE, width=1, dash="dot"), opacity=0.5, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit x", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_oracle, mode="lines", name=f"Orakel dieser Stichprobe (λ = {lam_text(a.lam_oracle)})", line=dict(color=C.COLOR_GREY, width=2, dash="dash"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_rule, mode="lines+markers", name=f"{C.RULE_SHORT[a.settings.rule]} (λ = {lam_text(a.lam_rule)})",
                             line=dict(color=C.RULE_COLORS[a.settings.rule], width=2.5), marker=dict(size=5), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    peak = float(np.max(np.abs(a.x_naive))) if np.all(np.isfinite(a.x_naive)) else float("inf")
    note = ""
    if peak > max(abs(lo), abs(hi)):
        pt = "unendlich" if not np.isfinite(peak) else (de(peak, 1) if peak < 1000.0 else sci(peak))
        note = f"Die direkte Inversion reicht bis ±{pt} und wird bei {de(lo, 1)} bis {de(hi, 1)} abgeschnitten."
    return _fixed(fig, 380, title=dict(text=note, font=dict(size=12, color="#9aa0a6"), x=0.0))


def build_error_chart(a: Analysis) -> go.Figure:
    """Tatsächlicher Fehler dieser Stichprobe und erwarteter Fehler über λ; markiert die drei Regeln, das Orakel und λ*."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.lams, y=a.exp_curve, mode="lines", name="erwarteter Fehler", line=dict(color="#222222", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=a.lams, y=a.err_curve, mode="lines", name="Fehler dieser Stichprobe", line=dict(color="#222222", width=3)))
    for key in ("dp", "gcv", "lcurve", "oracle"):
        i = a.picks[key]
        fig.add_trace(go.Scatter(x=[a.lams[i]], y=[a.err_curve[i]], mode="markers", name=f"{C.RULE_SHORT[key]}: {de(100 * a.err_curve[i], 1)} %",
                                 marker=dict(size=13 if key != a.settings.rule else 17, color=C.RULE_COLORS[key], symbol="diamond" if key == "oracle" else "circle",
                                             line=dict(width=2, color="#ffffff"))))
    _vline(fig, a.lam_star, "λ*", C.COLOR_GREY, "top right", "dash")
    _lam_axis(fig)
    fig.update_yaxes(title="Fehler relativ zur Signalnorm", type="log", range=[-2, 1.5], tickvals=[0.01, 0.1, 1, 10, 30], ticktext=["1 %", "10 %", "100 %", "1000 %", "3000 %"])
    return _fixed(fig, 400)


def build_gcv_chart(a: Analysis) -> go.Figure:
    i = a.picks["gcv"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.lams, y=np.maximum(a.gcv, LOG_FLOOR), mode="lines", name="G(λ) = ‖Ax − b‖² / (n − df)²", line=dict(color=C.COLOR_GCV, width=3)))
    fig.add_trace(go.Scatter(x=[a.lams[i]], y=[a.gcv[i]], mode="markers", name="Minimum", marker=dict(size=13, color=C.COLOR_GCV, line=dict(width=2, color="#ffffff"))))
    _vline(fig, a.lam_star, "λ*", C.COLOR_GREY, "top right", "dash")
    _lam_axis(fig)
    fig.update_yaxes(title="GCV-Funktion", type="log", tickformat=".0e")
    return _fixed(fig, 360)


def build_residual_chart(a: Analysis) -> go.Figure:
    i = a.picks["dp"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.lams, y=np.maximum(a.res, LOG_FLOOR), mode="lines", name="Residuum ‖A xλ − b‖", line=dict(color=C.COLOR_DP, width=3)))
    _ref_line(fig, a.target, f"Ziel τ·σ·√n (σ angenommen = {de(a.settings.sfactor, 2)}·σ)", "top left", "y", line=dict(color="#888888", dash="dash"))
    fig.add_trace(go.Scatter(x=[a.lams[i]], y=[a.res[i]], mode="markers", name="gewähltes λ", marker=dict(size=13, color=C.COLOR_DP, line=dict(width=2, color="#ffffff"))))
    _vline(fig, a.lam_star, "λ*", C.COLOR_GREY, "top right", "dash")
    _lam_axis(fig)
    fig.update_yaxes(title="Residuum", type="log", tickformat=".0e")
    return _fixed(fig, 360)


def build_lcurve_chart(a: Analysis) -> go.Figure:
    i = a.picks["lcurve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.maximum(a.res, LOG_FLOOR), y=a.sol, mode="lines", name="L-Kurve", line=dict(color=C.COLOR_LC, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[a.res[i]], y=[a.sol[i]], mode="markers", name=f"Ecke (λ = {lam_text(a.lams[i])})", marker=dict(size=13, color=C.COLOR_LC, line=dict(width=2, color="#ffffff"))))
    j = int(np.argmin(np.abs(np.log(a.lams / a.lam_star))))
    fig.add_trace(go.Scatter(x=[a.res[j]], y=[a.sol[j]], mode="markers", name=f"λ* = {lam_text(a.lam_star)}", marker=dict(size=12, color=C.COLOR_GREY, symbol="x", line=dict(width=2))))
    fig.update_xaxes(title="Residuum ‖A xλ − b‖", type="log", tickformat=".0e")
    fig.update_yaxes(title="Lösungsnorm ‖xλ‖", type="log", tickformat=".0e")
    return _fixed(fig, 360)


def build_box_chart(res: dict, kind: str) -> go.Figure:
    """Verteilung des Fehlers über die feste Wiederholungen je Regel (logarithmisch), mit dem erwarteten Optimum als Linie."""
    d = res["kinds"][kind]
    fig = go.Figure()
    for key in ("dp", "gcv", "lcurve", "oracle"):
        fig.add_trace(go.Box(y=d["rules"][key]["errors"], name=C.RULE_SHORT[key], marker_color=C.RULE_COLORS[key], boxpoints="outliers"))
    _ref_line(fig, d["err_star"], "erwartetes Optimum", "top left", "y", line=dict(color="#888888", dash="dash"))
    _ref_line(fig, C.FAIL_FACTOR * d["err_star"], "Doppeltes: gescheitert", "top left", "y", line=dict(color=C.COLOR_NAIVE, dash="dot"))
    fig.update_yaxes(title="Fehler relativ zur Signalnorm", type="log", tickvals=[0.05, 0.1, 0.2, 0.5, 1, 2, 5], ticktext=["5 %", "10 %", "20 %", "50 %", "100 %", "200 %", "500 %"])
    return _fixed(fig, 400, showlegend=False)


def build_fail_chart(rows: list[dict], kind: str) -> go.Figure:
    xs = [noise_label(r["noise"]) for r in rows]
    fig = go.Figure()
    for key in ("dp", "gcv", "lcurve"):
        fig.add_trace(go.Bar(x=xs, y=[100.0 * r[kind][key]["fail"] for r in rows], name=C.RULE_SHORT[key], marker_color=C.RULE_COLORS[key]))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Rauschen (% des Effektivwerts der Daten)", type="category")
    fig.update_yaxes(title="Anteil gescheitert (%)", range=[0, 105])
    return _fixed(fig, 340)


def build_sigma_chart(res: dict) -> go.Figure:
    rows = res["rows"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["factor"] for r in rows], y=[100.0 * r["rms"] for r in rows], mode="lines+markers", name="Diskrepanzprinzip, RMS-Fehler",
                             line=dict(color=C.COLOR_DP, width=2.5)))
    fig.add_hline(y=100.0 * res["err_star"], line=dict(color="#888888", dash="dash"), annotation_text="erwartetes Optimum", annotation_position="top left")
    fig.update_xaxes(title="Angenommenes σ als Vielfaches des wahren", type="log", tickvals=[r["factor"] for r in rows], ticktext=[de(r["factor"], 2).rstrip("0").rstrip(",") for r in rows])
    fig.update_yaxes(title="Fehler (% der Signalnorm)", range=[0, max(100.0 * max(r["rms"] for r in rows) * 1.1, 25.0)])
    return _fixed(fig, 360, showlegend=False)


def build_other_chart(res: dict) -> go.Figure:
    labels = [f"{C.OPERATOR_LABELS[r['op']].split(' (')[0]}, {noise_label(r['noise'])}" for r in res["ops"]]
    fig = go.Figure()
    for key in ("dp", "gcv", "lcurve"):
        fig.add_trace(go.Bar(x=labels, y=[100.0 * r[key]["fail"] for r in res["ops"]], name=C.RULE_SHORT[key], marker_color=C.RULE_COLORS[key]))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Matrix und Rauschen", type="category")
    fig.update_yaxes(title="Anteil gescheitert (%)", range=[0, 105])
    return _fixed(fig, 340)
