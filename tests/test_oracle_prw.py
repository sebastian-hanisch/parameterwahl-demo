"""Orakel (DEMO-PLAYBOOK §12): Kernroutinen gegen einen anderen Rechenweg (explizite Hat-Matrix, lstsq, scipy brentq und minimize_scalar, Monte Carlo).

Jeder Test läuft in unter 10 s. Fehlt scipy im Test-venv, überspringen sich die Tests (pytest.importorskip), sie raten nie."""

import math

import numpy as np
import pytest

import prw_model as model
from inverse_generator import SplitMix64, blurred_exact, draw_data, grid, noise_sigma, operator, phantom

scipy = pytest.importorskip("scipy")
from scipy import optimize  # noqa: E402

LAMS = np.array([1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 1.0])


def _setup(n=48, w=0.03, noise=1e-3, seed=5):
    a = operator(n, w)
    u, sv, vt = model.decompose(a)
    b = draw_data(n, w, noise, SplitMix64(seed), crime=False)[0]
    return a, u, sv, vt, b


@pytest.mark.parametrize("n,w", [(24, 0.02), (48, 0.03), (64, 0.03)])
def test_residual_solution_norm_and_df_match_the_explicit_formulas(n, w):
    a, u, sv, vt, b = _setup(n, w)
    cv = model.curves(u, sv, b[None, :], LAMS)
    for i, lam in enumerate(LAMS):
        xl = np.linalg.lstsq(np.vstack([a, lam * np.eye(n)]), np.concatenate([b, np.zeros(n)]), rcond=None)[0]
        hat = a @ np.linalg.solve(a.T @ a + lam ** 2 * np.eye(n), a.T)
        assert cv["res2"][0][i] == pytest.approx(float(np.sum((a @ xl - b) ** 2)), rel=1e-5, abs=1e-14)
        assert cv["sol2"][0][i] == pytest.approx(float(np.sum(xl ** 2)), rel=1e-6)
        assert cv["df"][i] == pytest.approx(float(np.trace(hat)), rel=1e-6)


def test_gcv_function_matches_the_definition_with_the_explicit_hat_matrix():
    n = 48
    a, u, sv, vt, b = _setup(n)
    cv = model.curves(u, sv, b[None, :], LAMS)
    g_ours = cv["res2"][0] / (n - cv["df"]) ** 2
    for i, lam in enumerate(LAMS):
        hat = a @ np.linalg.solve(a.T @ a + lam ** 2 * np.eye(n), a.T)
        ref = float(np.sum(((np.eye(n) - hat) @ b) ** 2)) / (n - float(np.trace(hat))) ** 2
        assert g_ours[i] == pytest.approx(ref, rel=1e-5)


def test_discrepancy_choice_matches_brentq_on_the_continuous_residual():
    n = 48
    a, u, sv, vt, b = _setup(n)
    sigma = noise_sigma(n, 0.03, 1e-3)
    beta = b @ u
    target = sigma ** 2 * n

    def res2(lam):
        f = sv ** 2 / (sv ** 2 + lam ** 2)
        return float(np.sum((1.0 - f) ** 2 * beta ** 2)) - target

    lam_ref = optimize.brentq(res2, 1e-8, 10.0)
    lams = model.lambda_grid()
    cv = model.curves(u, sv, b[None, :], lams)
    i = int(model.discrepancy_index(cv["res2"], sigma, n)[0])
    assert lams[i] <= lam_ref < lams[min(i + 1, len(lams) - 1)] + 1e-15              # die Gitterstelle ist die größte unterhalb der exakten Nullstelle


def test_residual_is_monotone_in_lambda():
    a, u, sv, vt, b = _setup()
    lams = model.lambda_grid()
    cv = model.curves(u, sv, b[None, :], lams)
    assert np.all(np.diff(cv["res2"][0]) >= -1e-18) and np.all(np.diff(cv["sol2"][0]) <= 1e-12)


def test_errors_match_the_error_of_an_independent_solution():
    n = 48
    a, u, sv, vt, b = _setup(n)
    x = phantom(grid(n))
    cv = model.curves(u, sv, b[None, :], LAMS)
    e = model.errors(vt, sv, cv["beta"], LAMS, vt @ x, float(np.linalg.norm(x)))[0]
    for i, lam in enumerate(LAMS):
        xl = np.linalg.solve(a.T @ a + lam ** 2 * np.eye(n), a.T @ b)
        assert e[i] == pytest.approx(float(np.linalg.norm(xl - x) / np.linalg.norm(x)), rel=1e-6)


@pytest.mark.parametrize("honest", [False, True])
def test_expected_errors_match_a_monte_carlo(honest):
    n, w, noise, lam = 48, 0.03, 1e-3, 5e-3
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    sigma = noise_sigma(n, w, noise)
    rho = (u.T @ (blurred_exact(grid(n), w) - a @ x)) / sv if honest else np.zeros(n)
    b = draw_data(n, w, noise, SplitMix64(11), crime=not honest, reps=3000)
    cv = model.curves(u, sv, b, np.array([lam]))
    err = model.errors(vt, sv, cv["beta"], np.array([lam]), vt @ x, float(np.linalg.norm(x)))[:, 0]
    sim = math.sqrt(float(np.mean(err ** 2)))
    ref = model.expected_errors(sv, vt @ x, rho, sigma, np.array([lam]), float(np.linalg.norm(x)))[0]
    assert sim == pytest.approx(ref, rel=0.02)


def test_best_expected_lambda_matches_scipy_bounded_minimization():
    n, w = 48, 0.03
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    c, xn = vt @ x, float(np.linalg.norm(x))
    rho = (u.T @ (blurred_exact(grid(n), w) - a @ x)) / sv
    sigma = noise_sigma(n, w, 1e-3)
    lams = model.lambda_grid()
    e = model.expected_errors(sv, c, rho, sigma, lams, xn)
    res = optimize.minimize_scalar(lambda t: float(model.expected_errors(sv, c, rho, sigma, np.array([10.0 ** t]), xn)[0]), bounds=(-8.0, 1.0), method="bounded",
                                   options={"xatol": 1e-10})
    assert float(e.min()) == pytest.approx(float(res.fun), rel=2e-3) and float(e.min()) >= float(res.fun) - 1e-12


def test_the_realization_oracle_never_loses_to_any_rule_and_branches_differ():
    n, w, noise = 48, 0.03, 1e-3
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    sigma = noise_sigma(n, w, noise)
    lams = model.lambda_grid()
    b = draw_data(n, w, noise, SplitMix64(21), crime=False, reps=200)
    cv = model.curves(u, sv, b, lams)
    err = model.errors(vt, sv, cv["beta"], lams, vt @ x, float(np.linalg.norm(x)))
    picks = {"dp": model.discrepancy_index(cv["res2"], sigma, n), "gcv": model.gcv_index(cv["res2"], cv["df"], n), "lcurve": model.lcurve_index(cv["res2"], cv["sol2"], lams),
             "oracle": model.oracle_index(err)}
    rows = np.arange(200)
    for key in ("dp", "gcv", "lcurve"):
        assert np.all(err[rows, picks["oracle"]] <= err[rows, picks[key]] + 1e-15)
    assert not np.array_equal(picks["dp"], picks["gcv"]) and not np.array_equal(picks["dp"], picks["lcurve"]) and not np.array_equal(picks["gcv"], picks["lcurve"])
