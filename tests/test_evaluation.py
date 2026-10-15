"""Auswertung: Formatierung, Analyse der Instanz, Verdikte (Art und Text), Form der Experimente und ihre Reproduzierbarkeit."""

import numpy as np
import pytest

import prw_constants as C
import prw_evaluation as ev


# ----------------------------------------------------------------------------- Formatierung

def test_de_sci_and_share():
    assert ev.de(-1.234, 2) == "−1,23" and ev.sci(2.5e9) == "2,5·10⁹" and ev.sci(3.0e-12) == "3,0·10⁻¹²" and ev.share(0.1525) == "15 %"


def test_err_pct_digits_by_magnitude():
    assert [ev.err_pct(v) for v in (2.58, 0.1188, 0.0399, 0.0087, 8.7e-4, 4471.0, 0.57)] == ["258 %", "11,9 %", "4,0 %", "0,87 %", "0,09 %", "4,5·10³-fach", "57,0 %"]


def test_noise_label_and_lam_text():
    assert [ev.noise_label(v) for v in C.NOISE_CHOICES] == ["0 %", "0,0001 %", "0,001 %", "0,01 %", "0,1 %", "1 %", "10 %"]
    assert ev.lam_text(0.0094) == "9,4·10⁻³" and ev.lam_text(0.5) == "0,500" and ev.lam_text(0.0) == "0"


# ----------------------------------------------------------------------------- Analyse der Instanz

def test_the_default_settings_are_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_picks_are_consistent_with_the_chosen_rule():
    for rule in C.RULES:
        a = ev.analyse(ev.Settings(rule=rule))
        assert a.lam_rule == pytest.approx(float(a.lams[a.picks[rule]])) and a.err_rule == pytest.approx(a.pick_errors[rule])
        assert a.err_oracle <= min(a.pick_errors[k] for k in ("dp", "gcv", "lcurve")) + 1e-15


def test_analysis_is_deterministic_and_the_seed_matters():
    a, b, c = ev.analyse(ev.Settings()), ev.analyse(ev.Settings()), ev.analyse(ev.Settings(seed=12))
    assert np.array_equal(a.x_rule, b.x_rule) and a.picks == b.picks and not np.array_equal(a.b, c.b)


def test_the_oracle_error_is_the_minimum_of_the_error_curve():
    a = ev.analyse(ev.Settings())
    assert a.err_oracle == pytest.approx(float(a.err_curve.min())) and a.err_curve.shape == a.lams.shape == a.exp_curve.shape


def test_the_discrepancy_target_scales_with_the_assumed_sigma():
    a1 = ev.analyse(ev.Settings(sfactor=1.0))
    a4 = ev.analyse(ev.Settings(sfactor=4.0))
    assert a4.target == pytest.approx(4.0 * a1.target) and a4.lam_rule > a1.lam_rule


def test_a_failed_sample_is_flagged_by_the_double_optimum_rule():
    a = ev.analyse(ev.Settings(**C.PRESETS["GCV scheitert"]))
    assert a.failed and a.err_rule > C.FAIL_FACTOR * a.err_star
    b = ev.analyse(ev.Settings())
    assert not b.failed


# ----------------------------------------------------------------------------- Verdikte

def test_verdict_kinds_and_texts_of_the_presets():
    out = {name: ev.verdict_instance(ev.analyse(ev.Settings(**p))) for name, p in C.PRESETS.items()}
    assert out["Standard"][0] == "success" and all(out[k][0] == "warning" for k in ("GCV scheitert", "L-Kurve zu klein", "σ unterschätzt"))
    assert "zu klein" in out["GCV scheitert"][1] and "Orakel" in out["Standard"][1]


# ----------------------------------------------------------------------------- Experimente: Form und Reproduzierbarkeit

def test_who_experiment_structure_and_reproducibility():
    r1, r2 = ev.who_experiment(reps=60), ev.who_experiment(reps=60)
    assert set(r1["kinds"]) == {"honest", "crime"} and set(r1["kinds"]["honest"]["rules"]) == {"dp", "gcv", "lcurve", "oracle"}
    assert np.array_equal(r1["kinds"]["honest"]["rules"]["gcv"]["errors"], r2["kinds"]["honest"]["rules"]["gcv"]["errors"])
    o = r1["kinds"]["honest"]["rules"]["oracle"]
    assert o["fail"] == 0.0 and o["lam_ratio"] == pytest.approx(1.0, abs=0.15)


def test_the_discrepancy_principle_never_fails_on_the_standard_instance_but_the_others_do():
    r = ev.who_experiment()["kinds"]["honest"]["rules"]
    assert r["dp"]["fail"] == 0.0 and r["gcv"]["fail"] > 0.05 and r["lcurve"]["fail"] > 0.3
    assert r["gcv"]["median"] < 1.2 * r["oracle"]["median"] < r["lcurve"]["median"]


def test_noise_experiment_rows_and_shape():
    rows = ev.noise_experiment(levels=(1e-2, 1e-3), reps=100)
    assert [r["noise"] for r in rows] == [1e-2, 1e-3] and all(set(r) == {"noise", "honest", "crime"} for r in rows)
    assert rows[0]["honest"]["dp"]["median"] >= rows[1]["honest"]["dp"]["median"] - 0.02


def test_grid_experiment_coarse_grids_need_no_penalty():
    rows = {r["n"]: r for r in ev.grid_experiment(ns=(24, 48), reps=100)}
    assert rows[24]["lam_star"] < 1e-6 and rows[48]["lam_star"] > 1e-3
    assert abs(rows[24]["gcv"]["median"] - rows[24]["oracle"]["median"]) < 0.01


def test_sigma_experiment_underestimating_hurts_more_than_overestimating():
    res = ev.sigma_experiment(reps=150)
    by = {r["factor"]: r for r in res["rows"]}
    assert by[0.25]["rms"] > 3 * by[1.0]["rms"] and by[4.0]["rms"] < 1.15 * by[1.0]["rms"]
    assert [t["tau"] for t in res["taus"]] == [1.0, 1.1, 1.5, 2.0]
    tail = {(t["noise"], t["kind"]): t for t in res["tail"]}
    assert tail[(1e-3, "honest")]["ratio"] == pytest.approx(1.0, abs=0.15) and tail[(1e-5, "honest")]["ratio"] > 3.0


def test_other_experiment_structure():
    res = ev.other_experiment(reps=100)
    assert len(res["ops"]) == len(C.EXP_OPERATORS) * len(C.EXP_OP_NOISE) and [b["kind"] for b in res["bounded"]] == ["honest", "crime"]
    assert [r["factor"] for r in res["bounded"][0]["rows"]] == [0.0, *C.GCV_BOUND_FACTORS]


def test_ratio_text_switches_to_powers_of_ten_below_one_hundredth():
    assert ev.ratio_text(1.884) == "1,88" and ev.ratio_text(0.06) == "0,06" and ev.ratio_text(1.2e-6) == "1,2·10⁻⁶"
