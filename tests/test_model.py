"""Kern: jede Einheit an der 3 x 3-Mini-Instanz von Hand (tests/conftest.py) und an Eigenschaften der Kurven und Regeln."""

import math

import numpy as np
import pytest

import prw_constants as C
import prw_model as model
from inverse_generator import grid, operator, phantom

LAMS_MINI = np.array([0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0])


def _mini_curves(mini, b=None):
    a, x, bb = mini
    u, sv, vt = model.decompose(a)
    return u, sv, vt, model.curves(u, sv, (bb if b is None else b)[None, :], LAMS_MINI)


def test_the_grid_has_the_documented_bounds_and_size():
    g = model.lambda_grid()
    lo, hi, pts = C.LAM_GRID
    assert len(g) == pts and g[0] == pytest.approx(lo) and g[-1] == pytest.approx(hi) and np.all(np.diff(np.log(g)) > 0)


def test_mini_filter_matrix_by_hand(mini):
    a, *_ = mini
    _, sv, _ = model.decompose(a)
    f = model.filter_matrix(sv, np.array([0.0, 0.1]))
    assert f[0].tolist() == pytest.approx([1.0, 1.0, 1.0]) and f[1] == pytest.approx([1.0 / 1.01, 0.25 / 0.26, 0.5], abs=1e-12)


def test_mini_curves_by_hand(mini):
    u, sv, vt, cv = _mini_curves(mini)
    assert cv["beta"][0] == pytest.approx([-4.0 / math.sqrt(3.0), 0.0, -0.2 / math.sqrt(6.0)], abs=1e-12)
    idx = {0.01: 0, 0.05: 2, 0.1: 4, 0.5: 6}
    res2 = {0.01: 7.06853e-7, 0.05: 2.99834e-4, 0.1: 2.18949e-3, 0.5: 0.219497}
    sol2 = {0.01: 5.9858, 0.05: 5.7334, 0.1: 5.3949, 0.5: 3.4143}
    df = {0.01: 2.9896, 0.05: 2.7876, 0.1: 2.4516, 0.5: 1.3385}
    for lam, i in idx.items():
        assert cv["res2"][0][i] == pytest.approx(res2[lam], rel=1e-4)
        assert cv["sol2"][0][i] == pytest.approx(sol2[lam], rel=1e-4)
        assert cv["df"][i] == pytest.approx(df[lam], abs=1e-4)


def test_mini_residual_formula_by_hand(mini):
    u, sv, vt, cv = _mini_curves(mini)
    for i, lam in enumerate(LAMS_MINI):
        hand = (lam ** 2 / (1.0 + lam ** 2)) ** 2 * (16.0 / 3.0) + (lam ** 2 / (0.01 + lam ** 2)) ** 2 * (0.04 / 6.0)
        assert cv["res2"][0][i] == pytest.approx(hand, rel=1e-9)


def test_mini_discrepancy_principle_by_hand(mini):
    u, sv, vt, cv = _mini_curves(mini)
    i = int(model.discrepancy_index(cv["res2"], 0.01, 3)[0])
    assert LAMS_MINI[i] == pytest.approx(0.05)                            # Residuum^2(0,05) = 2,998e-4 <= 3e-4 < Residuum^2(0,07)
    # tau = 2: Ziel (2 * 0,01)^2 * 3 = 1,2e-3; Residuum^2(0,07) = 8,48e-4 liegt darunter, Residuum^2(0,1) = 2,19e-3 darüber
    assert LAMS_MINI[int(model.discrepancy_index(cv["res2"], 0.01, 3, tau=2.0)[0])] == pytest.approx(0.07)


def test_mini_discrepancy_with_no_feasible_lambda_returns_the_smallest(mini):
    u, sv, vt, cv = _mini_curves(mini)
    assert int(model.discrepancy_index(cv["res2"], 1e-6, 3)[0]) == 0


def test_discrepancy_accepts_a_vector_of_sigmas(mini):
    u, sv, vt, cv = _mini_curves(mini)
    res2 = np.vstack([cv["res2"][0], cv["res2"][0]])
    idx = model.discrepancy_index(res2, np.array([0.01, 0.1]), 3)
    assert LAMS_MINI[idx[0]] == pytest.approx(0.05) and idx[1] > idx[0]


def test_mini_gcv_of_noise_free_data_picks_the_smallest_lambda(mini):
    u, sv, vt, cv = _mini_curves(mini)
    g = cv["res2"][0] / (3 - cv["df"]) ** 2
    assert g[0] == pytest.approx(0.00653423, abs=1e-7) and g[2] == pytest.approx(0.0066465, abs=1e-6) and g[4] == pytest.approx(0.00728128, abs=1e-6)
    assert int(model.gcv_index(cv["res2"], cv["df"], 3)[0]) == 0 and np.all(np.diff(g) > 0)


def test_gcv_bounded_respects_the_lower_bound(mini):
    u, sv, vt, cv = _mini_curves(mini)
    i = int(model.gcv_index_bounded(cv["res2"], cv["df"], 3, LAMS_MINI, 0.05)[0])
    assert LAMS_MINI[i] == pytest.approx(0.05)


def test_lcurve_curvature_of_a_circle_is_one_over_the_radius():
    r = 1.7
    t = np.linspace(-1.2, 1.2, 241)
    lams = np.exp(t)
    res2 = np.exp(2.0 * r * np.cos(t))
    sol2 = np.exp(2.0 * r * np.sin(t))
    k = model.lcurve_curvature(res2[None, :], sol2[None, :], lams)[0]
    assert k[5:-5] == pytest.approx(np.full(len(k) - 10, 1.0 / r), rel=0.02)


def test_lcurve_index_finds_the_corner_of_an_l_shape():
    """Weiche L-Form: log Residuum ~ softplus(3 t)/3 (flach links, steigend rechts), log Lösungsnorm ~ softplus(-3 t)/3 (fallend links, flach rechts); Ecke bei t = 0."""
    t = np.linspace(-4.0, 4.0, 801)
    lams = np.exp(t)
    rho = np.logaddexp(0.0, 3.0 * t) / 3.0
    eta = np.logaddexp(0.0, -3.0 * t) / 3.0
    res2, sol2 = np.exp(2.0 * rho), np.exp(2.0 * eta)
    i = int(model.lcurve_index(res2[None, :], sol2[None, :], lams)[0])
    assert abs(t[i]) < 0.1


def test_oracle_index_is_the_row_argmin():
    err = np.array([[0.5, 0.2, 0.9], [0.1, 0.3, 0.05]])
    assert model.oracle_index(err).tolist() == [1, 2]


def test_noise_tail_estimate_by_hand():
    beta = np.array([[9.0, 9.0, 1.0, -1.0, 2.0, -2.0]])
    assert model.noise_tail_estimate(beta, 4)[0] == pytest.approx(math.sqrt((1 + 1 + 4 + 4) / 4.0))


def test_errors_match_a_direct_solve(mini):
    a, x, b = mini
    u, sv, vt = model.decompose(a)
    cv = model.curves(u, sv, b[None, :], LAMS_MINI)
    e = model.errors(vt, sv, cv["beta"], LAMS_MINI, vt @ x, float(np.linalg.norm(x)))[0]
    for i, lam in enumerate(LAMS_MINI):
        xl = np.linalg.solve(a.T @ a + lam ** 2 * np.eye(3), a.T @ b)
        assert e[i] == pytest.approx(float(np.linalg.norm(xl - x) / np.linalg.norm(x)), rel=1e-9)


def test_expected_errors_at_lambda_zero_without_noise_vanish(mini):
    a, x, b = mini
    u, sv, vt = model.decompose(a)
    e = model.expected_errors(sv, vt @ x, np.zeros(3), 0.0, np.array([1e-12, 0.1]), float(np.linalg.norm(x)))
    assert e[0] == pytest.approx(0.0, abs=1e-9) and e[1] == pytest.approx(0.1669, abs=1e-4)


def test_integration_matrices_by_hand():
    assert model.integration_matrix(3, 1) == pytest.approx(np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]]) / 3.0)
    assert model.integration_matrix(3, 2) == pytest.approx(np.array([[1, 0, 0], [2, 1, 0], [3, 2, 1]]) / 9.0)


def test_the_expected_best_lambda_sits_in_the_interior_for_the_standard_instance():
    n = 48
    a = operator(n, 0.03)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    lams = model.lambda_grid()
    e = model.expected_errors(sv, vt @ x, np.zeros(n), 3.4e-4, lams, float(np.linalg.norm(x)))
    i = int(np.argmin(e))
    assert 0 < i < len(lams) - 1 and 1e-3 < lams[i] < 1e-1


def test_gcv_curve_is_finite_when_df_reaches_n(mini):
    a, x, b = mini
    u, sv, vt = model.decompose(a)
    cv = model.curves(u, sv, b[None, :], np.array([1e-12, 1e-9, 0.1]))
    g = model.gcv_curve(cv["res2"], cv["df"], 3)
    assert np.all(np.isfinite(g)) and np.all(g >= 0.0)
    assert g[0, 2] == pytest.approx(cv["res2"][0, 2] / (3 - cv["df"][2]) ** 2, rel=1e-12)


def test_discrepancy_accepts_a_residual_exactly_at_the_target():
    res2 = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert int(model.discrepancy_index(res2, 1.0, 3)[0]) == 2          # Ziel sigma^2 n = 3: Gleichheit gilt als erfüllt
    assert int(model.discrepancy_index(res2, 1.0, 3, tau=1.0001)[0]) == 2
    assert int(model.discrepancy_index(res2, 0.99, 3)[0]) == 1
