"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über feste Zufallsströme (prw_constants.SEED_BASE), nie über einen einzelnen Seed als Befund; die Einzelstichproben der Presets sind
ausdrücklich einzelne Ströme. Kennzahlen aus 400 Wiederholungen tragen die Toleranz ihres Stichprobenfehlers (Anteile ±3 Punkte, Fehler ±0,003).
"""

import numpy as np
import pytest

import prw_constants as C
import prw_evaluation as ev
import prw_model as model


@pytest.fixture(scope="module")
def who():
    return ev.who_experiment()


@pytest.fixture(scope="module")
def noise_rows():
    return {r["noise"]: r for r in ev.noise_experiment()}


@pytest.fixture(scope="module")
def grid_rows():
    return {r["n"]: r for r in ev.grid_experiment()}


@pytest.fixture(scope="module")
def sigma():
    return ev.sigma_experiment()


@pytest.fixture(scope="module")
def other():
    return ev.other_experiment()


# ----------------------------------------------------------------------------- Einzelstichproben

def test_standard_sample_and_presets():
    a = ev.analyse(ev.Settings())
    assert a.lam_star == pytest.approx(0.00944, rel=0.01) and a.err_star == pytest.approx(0.1168, abs=0.0005) and a.err_naive == pytest.approx(2.37, abs=0.02)
    assert a.err_rule == pytest.approx(0.1183, abs=0.001) and a.lam_rule / a.lam_star == pytest.approx(1.88, abs=0.03) and a.err_oracle == pytest.approx(0.1175, abs=0.001)
    g = ev.analyse(ev.Settings(**C.PRESETS["GCV scheitert"]))
    assert g.err_rule == pytest.approx(0.856, abs=0.005) and g.lam_rule / g.lam_star < 0.005
    lc = ev.analyse(ev.Settings(**C.PRESETS["L-Kurve zu klein"]))
    assert lc.err_rule == pytest.approx(0.317, abs=0.003) and lc.lam_rule / lc.lam_star == pytest.approx(0.060, abs=0.003)
    sg = ev.analyse(ev.Settings(**C.PRESETS["σ unterschätzt"]))
    assert sg.err_rule == pytest.approx(0.828, abs=0.005) and sg.lam_rule / sg.lam_star == pytest.approx(0.022, abs=0.003)


# ----------------------------------------------------------------------------- Experiment 1

def test_experiment_1_honest(who):
    k = who["kinds"]["honest"]
    assert k["lam_star"] == pytest.approx(0.00944, rel=0.01) and k["err_star"] == pytest.approx(0.1168, abs=0.0005)
    dp, gcv, lc, orc = (k["rules"][r] for r in ("dp", "gcv", "lcurve", "oracle"))
    assert dp["median"] == pytest.approx(0.1188, abs=0.002) and dp["p90"] == pytest.approx(0.1211, abs=0.003) and dp["fail"] == 0.0 and dp["lam_ratio"] == pytest.approx(2.0, abs=0.1)
    assert gcv["median"] == pytest.approx(0.1183, abs=0.003) and gcv["p90"] == pytest.approx(0.634, abs=0.05) and gcv["fail"] == pytest.approx(0.15, abs=0.03)
    assert gcv["rms"] == pytest.approx(0.42, abs=0.05) and gcv["lam_ratio"] == pytest.approx(0.75, abs=0.05) and gcv["ratio_q"][0] < 0.01
    assert lc["median"] == pytest.approx(0.27, abs=0.01) and lc["fail"] == pytest.approx(0.68, abs=0.05) and lc["lam_ratio"] == pytest.approx(0.06, abs=0.01)
    assert orc["median"] == pytest.approx(0.1165, abs=0.002) and orc["fail"] == 0.0


def test_experiment_1_crime(who):
    k = who["kinds"]["crime"]
    assert k["lam_star"] == pytest.approx(0.00708, rel=0.01) and k["err_star"] == pytest.approx(0.0710, abs=0.0005)
    dp, gcv, lc, orc = (k["rules"][r] for r in ("dp", "gcv", "lcurve", "oracle"))
    assert dp["median"] == pytest.approx(0.0781, abs=0.002) and dp["lam_ratio"] == pytest.approx(2.37, abs=0.1) and dp["fail"] == 0.0
    assert gcv["median"] == pytest.approx(0.0747, abs=0.003) and gcv["fail"] == pytest.approx(0.205, abs=0.03) and gcv["p90"] == pytest.approx(0.87, abs=0.08)
    assert lc["median"] == pytest.approx(0.2375, abs=0.01) and lc["fail"] == pytest.approx(0.90, abs=0.05) and orc["median"] == pytest.approx(0.0703, abs=0.002)


def test_experiment_1_quantiles_of_the_chosen_parameter(who):
    r = who["kinds"]["honest"]["rules"]
    assert np.round(r["dp"]["ratio_q"], 1).tolist() == [1.8, 1.9, 2.0, 2.0, 2.1]
    assert np.round(r["oracle"]["ratio_q"], 1).tolist() == [0.5, 0.8, 1.0, 1.2, 1.8]


# ----------------------------------------------------------------------------- Experiment 2

def test_experiment_2_failure_shares_by_noise(noise_rows):
    f = lambda kind, rule: [round(100 * noise_rows[nz][kind][rule]["fail"]) for nz in C.EXP_NOISE_LEVELS]  # noqa: E731
    assert f("honest", "dp") == [0, 0, 0, 0, 0] and f("crime", "dp") == [0, 0, 0, 0, 99]
    gh, gc = f("honest", "gcv"), f("crime", "gcv")
    assert all(abs(a - b) <= 4 for a, b in zip(gh, [13, 16, 16, 3, 0])) and all(abs(a - b) <= 5 for a, b in zip(gc, [14, 16, 21, 16, 65]))
    lh, lcrime = f("honest", "lcurve"), f("crime", "lcurve")
    assert all(abs(a - b) <= 5 for a, b in zip(lh, [25, 71, 60, 0, 0])) and all(abs(a - b) <= 5 for a, b in zip(lcrime, [28, 85, 92, 0, 90]))


def test_experiment_2_median_errors_by_noise(noise_rows):
    h = {nz: noise_rows[nz]["honest"] for nz in noise_rows}
    assert h[1e-1]["dp"]["median"] == pytest.approx(0.194, abs=0.003) and h[1e-2]["dp"]["median"] == pytest.approx(0.135, abs=0.003)
    assert h[1e-3]["lcurve"]["median"] == pytest.approx(0.256, abs=0.01) and h[1e-5]["dp"]["median"] == pytest.approx(0.117, abs=0.003)
    c5 = noise_rows[1e-5]["crime"]
    assert c5["err_star"] == pytest.approx(0.0223, abs=0.001) and c5["dp"]["median"] == pytest.approx(0.0545, abs=0.003) and c5["oracle"]["median"] == pytest.approx(0.0171, abs=0.002)


def test_experiment_2_grid(grid_rows):
    assert grid_rows[24]["lam_star"] < 1e-6 and grid_rows[32]["lam_star"] < 1e-6
    assert grid_rows[48]["gcv"]["fail"] == pytest.approx(0.18, abs=0.04) and grid_rows[64]["gcv"]["fail"] == pytest.approx(0.14, abs=0.04)
    assert grid_rows[96]["gcv"]["fail"] == pytest.approx(0.03, abs=0.03) and grid_rows[64]["gcv"]["rms"] > 50.0
    assert grid_rows[64]["dp"]["median"] == pytest.approx(0.0963, abs=0.002) and grid_rows[96]["dp"]["median"] == pytest.approx(0.1022, abs=0.002)
    assert grid_rows[48]["lcurve"]["fail"] == pytest.approx(0.64, abs=0.06) and grid_rows[96]["lcurve"]["fail"] == pytest.approx(0.12, abs=0.05)


# ----------------------------------------------------------------------------- Experiment 3

def test_experiment_3_assumed_sigma(sigma):
    by = {r["factor"]: r for r in sigma["rows"]}
    assert by[0.25]["rms"] == pytest.approx(0.82, abs=0.03) and by[0.25]["fail"] == pytest.approx(0.80, abs=0.05) and by[0.25]["lam_ratio"] == pytest.approx(0.034, abs=0.01)
    assert by[0.5]["rms"] == pytest.approx(0.169, abs=0.01) and by[0.5]["fail"] == pytest.approx(0.06, abs=0.03)
    assert by[1.0]["rms"] == pytest.approx(0.1186, abs=0.002) and by[2.0]["rms"] == pytest.approx(0.1218, abs=0.002) and by[4.0]["rms"] == pytest.approx(0.1261, abs=0.002)
    assert by[0.7]["rms"] == pytest.approx(0.118, abs=0.002)


def test_experiment_3_tau(sigma):
    assert [round(t["rms"], 3) for t in sigma["taus"]] == [0.119, 0.119, 0.120, 0.122]


def test_experiment_3_noise_estimate(sigma):
    t = {(r["noise"], r["kind"]): r for r in sigma["tail"]}
    assert [round(t[(nz, "honest")]["ratio"], 2) for nz in (1e-1, 1e-2, 1e-3)] == pytest.approx([0.95, 0.96, 0.96], abs=0.02)
    assert t[(1e-4, "honest")]["ratio"] == pytest.approx(1.25, abs=0.1) and t[(1e-5, "honest")]["ratio"] == pytest.approx(8.4, abs=0.5)
    assert t[(1e-1, "honest")]["dp_hat"]["rms"] == pytest.approx(0.50, abs=0.05) and t[(1e-1, "honest")]["dp_true"]["rms"] == pytest.approx(0.195, abs=0.01)
    assert t[(1e-3, "honest")]["dp_hat"]["rms"] == pytest.approx(0.1188, abs=0.003) and t[(1e-2, "honest")]["dp_hat"]["rms"] == pytest.approx(0.159, abs=0.01)


# ----------------------------------------------------------------------------- Experiment 4

def test_experiment_4_other_matrices(other):
    by = {(r["op"], r["noise"]): r for r in other["ops"]}
    assert by[("integral", 1e-2)]["gcv"]["rms"] == pytest.approx(0.27, abs=0.03) and by[("integral", 1e-2)]["lcurve"]["fail"] == pytest.approx(0.54, abs=0.06)
    assert by[("integral2", 1e-2)]["dp"]["rms"] == pytest.approx(0.468, abs=0.02) and by[("integral2", 1e-2)]["err_star"] == pytest.approx(0.401, abs=0.002)
    assert by[("integral2", 1e-2)]["gcv"]["fail"] == pytest.approx(0.13, abs=0.04) and by[("integral2", 1e-3)]["gcv"]["fail"] == pytest.approx(0.17, abs=0.04)
    assert max(r["dp"]["rms"] / r["err_star"] for r in other["ops"]) == pytest.approx(1.17, abs=0.03)


def test_experiment_4_bounded_gcv(other):
    h = {r["factor"]: r["fail"] for r in other["bounded"][0]["rows"]}
    c = {r["factor"]: r["fail"] for r in other["bounded"][1]["rows"]}
    assert h[0.0] == pytest.approx(0.14, abs=0.04) and h[1.0] == pytest.approx(0.11, abs=0.04) and h[10.0] == pytest.approx(0.025, abs=0.025)
    assert c[0.0] == pytest.approx(0.235, abs=0.04) and c[1.0] == pytest.approx(0.22, abs=0.04) and c[10.0] == pytest.approx(0.12, abs=0.04)


def test_mini_instance_by_hand_numbers(mini):
    a, x, b = mini
    u, sv, vt = model.decompose(a)
    lams = np.array([0.01, 0.03, 0.05, 0.07, 0.1])
    cv = model.curves(u, sv, b[None, :], lams)
    assert lams[int(model.discrepancy_index(cv["res2"], 0.01, 3)[0])] == pytest.approx(0.05)
    assert cv["df"][4] == pytest.approx(2.4516, abs=1e-4) and int(model.gcv_index(cv["res2"], cv["df"], 3)[0]) == 0


def test_gcv_failure_anatomy(who):
    a = who["kinds"]["honest"]["gcv_anatomy"]
    assert a["tail_failed"] == pytest.approx(0.81, abs=0.05) and a["tail_ok"] == pytest.approx(1.00, abs=0.03)
    assert a["naive_failed"] == pytest.approx(1.32, abs=0.1) and a["naive_all"] == pytest.approx(2.30, abs=0.1) and a["tiny_lambda"] == pytest.approx(0.56, abs=0.08)
