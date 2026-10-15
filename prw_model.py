"""Kern: Tikhonov-Kurven über einem λ-Gitter und die drei Regeln zur Parameterwahl ohne Wahrheit (GCV, Diskrepanzprinzip, L-Kurve), das Realisierungsorakel,
die geschlossene Erwartung und die Rauschschätzung. Nur numpy, kein Streamlit.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand und
`tests/test_oracle_prw.py` gegen die Definition mit expliziter Hat-Matrix, scipy (`brentq`), Aufzählung und Monte Carlo.

Bezeichnungen: A = U diag(sv) V^T, x die Wahrheit, c = V^T x, beta = U^T b die Komponenten der Daten, F[l, i] = sv_i^2 / (sv_i^2 + lam_l^2) die Filterfaktoren.
Zu jedem lam gehört die Tikhonov-Lösung x_lam = V diag(F / sv) beta; Residuum^2 = sum (1 - F)^2 beta^2, Lösungsnorm^2 = sum (F / sv)^2 beta^2, df = sum F.
"""

from __future__ import annotations

import math

import numpy as np

import prw_constants as C

_TINY = 1e-300


def lambda_grid() -> np.ndarray:
    lo, hi, pts = C.LAM_GRID
    return np.logspace(math.log10(lo), math.log10(hi), pts)


def decompose(a: np.ndarray):
    """Volle SVD A = U diag(sv) V^T (sv absteigend)."""
    u, sv, vt = np.linalg.svd(a)
    return u, sv, vt


def filter_matrix(sv: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """F[l, i] = sv_i^2 / (sv_i^2 + lam_l^2)."""
    s2 = np.asarray(sv, dtype=float)[None, :] ** 2
    return s2 / np.maximum(s2 + np.asarray(lams, dtype=float)[:, None] ** 2, _TINY)


def curves(u: np.ndarray, sv: np.ndarray, b: np.ndarray, lams: np.ndarray) -> dict:
    """Residuum^2, Lösungsnorm^2 und Freiheitsgrade über das λ-Gitter, je Zeile von b (Form (reps, n)); Rückgabe in Form (reps, L) bzw. (L,)."""
    beta = np.atleast_2d(b) @ u
    f = filter_matrix(sv, lams)
    res2 = ((1.0 - f) ** 2) @ (beta ** 2).T
    sol2 = ((f / np.maximum(sv, _TINY)) ** 2) @ (beta ** 2).T
    return {"res2": res2.T, "sol2": sol2.T, "df": f.sum(axis=1), "beta": beta}


def errors(vt: np.ndarray, sv: np.ndarray, beta: np.ndarray, lams: np.ndarray, c: np.ndarray, x_norm: float) -> np.ndarray:
    """Tatsächlicher relativer Fehler ||x_lam - x|| / ||x|| je Zeile von beta und je lam, Form (reps, L); V ist orthogonal, also zählen die Koeffizienten."""
    f = filter_matrix(sv, lams)
    out = np.empty((beta.shape[0], len(lams)))
    gain = f / np.maximum(sv, _TINY)
    for r in range(beta.shape[0]):
        out[r] = np.sqrt(((gain * beta[r][None, :] - c[None, :]) ** 2).sum(axis=1)) / x_norm
    return out


def expected_errors(sv, c, rho, sigma, lams, x_norm) -> np.ndarray:
    """Erwarteter Fehler (geschlossen) über das λ-Gitter: Bias f (c + rho) - c und Varianz f^2 sigma^2 / sv^2, rho = (U^T r) / sv (Modellfehler der Daten)."""
    f = filter_matrix(sv, lams)
    bias = f * (c + rho)[None, :] - c[None, :]
    var = f ** 2 * sigma ** 2 / np.maximum(sv, _TINY)[None, :] ** 2
    return np.sqrt((bias ** 2 + var).sum(axis=1)) / x_norm


# ----------------------------------------------------------------------------- die drei Regeln und das Orakel (Index im λ-Gitter je Zeile)

_GCV_FLOOR = 1e-12                                                 # n - df darf rundungsbedingt null werden; darunter wird nicht geteilt


def gcv_curve(res2: np.ndarray, df: np.ndarray, n: int) -> np.ndarray:
    """GCV-Funktion G(lam) = Residuum^2 / (n - df)^2 je Zeile von res2 (Form (reps, L))."""
    return res2 / np.maximum(n - df, _GCV_FLOOR)[None, :] ** 2


def gcv_index(res2: np.ndarray, df: np.ndarray, n: int) -> np.ndarray:
    """GCV: Minimum von G(lam) (kleinster Index bei Gleichstand)."""
    return np.argmin(gcv_curve(res2, df, n), axis=1)


def gcv_index_bounded(res2: np.ndarray, df: np.ndarray, n: int, lams: np.ndarray, lam_min: float) -> np.ndarray:
    """GCV nur für lam >= lam_min (Untergrenze gegen Fehlminima am Gitterrand)."""
    g = np.where((lams >= lam_min)[None, :], gcv_curve(res2, df, n), np.inf)
    return np.argmin(g, axis=1)


def discrepancy_index(res2: np.ndarray, sigma: float, n: int, tau: float = 1.0) -> np.ndarray:
    """Diskrepanzprinzip: größtes lam des Gitters mit Residuum^2 <= (tau sigma)^2 n; gibt es keines, der Index 0 (kleinstes lam).
    `sigma` darf ein Skalar oder ein Vektor je Zeile sein."""
    target = (tau * np.asarray(sigma, dtype=float)) ** 2 * n
    ok = res2 <= (target.reshape(-1, 1) if target.ndim else target)
    idx = (ok * np.arange(res2.shape[1])[None, :]).max(axis=1)
    return np.where(ok.any(axis=1), idx, 0)


def lcurve_curvature(res2: np.ndarray, sol2: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """Krümmung der Kurve (log Residuum, log Lösungsnorm) über log lam, je Zeile (endliche Differenzen auf dem Gitter)."""
    rho = 0.5 * np.log(np.maximum(res2, _TINY))
    eta = 0.5 * np.log(np.maximum(sol2, _TINY))
    t = np.log(lams)
    d1r, d1e = np.gradient(rho, t, axis=1), np.gradient(eta, t, axis=1)
    d2r, d2e = np.gradient(d1r, t, axis=1), np.gradient(d1e, t, axis=1)
    return (d1r * d2e - d2r * d1e) / np.maximum((d1r ** 2 + d1e ** 2) ** 1.5, _TINY)


def lcurve_index(res2: np.ndarray, sol2: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """L-Kurve: Stelle der größten Krümmung."""
    return np.argmax(lcurve_curvature(res2, sol2, lams), axis=1)


def oracle_index(err: np.ndarray) -> np.ndarray:
    """Realisierungsorakel: das lam mit dem kleinsten tatsächlichen Fehler der jeweiligen Zeile."""
    return np.argmin(err, axis=1)


def noise_tail_estimate(beta: np.ndarray, m: int = C.TAIL_M) -> np.ndarray:
    """Rauschschätzung je Zeile: Effektivwert der letzten m Komponenten |u_i^T b| (dort steckt, solange das Signal klein ist, nur Rauschen)."""
    return np.sqrt(np.mean(np.atleast_2d(beta)[:, -m:] ** 2, axis=1))


def integration_matrix(n: int, times: int = 1) -> np.ndarray:
    """Integrationsoperator h * (untere Dreiecksmatrix aus Einsen), `times`-fach angewendet."""
    v = np.tril(np.ones((n, n))) / n
    out = np.eye(n)
    for _ in range(times):
        out = v @ out
    return out
