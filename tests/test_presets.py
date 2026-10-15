"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen."""

import pytest

import prw_constants as C
import prw_evaluation as ev
import prw_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_on_the_step_grid():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0, name
        assert C.W_MIN <= p["w"] <= C.W_MAX and (p["w"] - C.W_MIN) / C.W_STEP == pytest.approx(round((p["w"] - C.W_MIN) / C.W_STEP)), name
        assert p["noise"] in C.NOISE_CHOICES and p["data"] in C.DATA_CHOICES and p["rule"] in C.RULES and p["sfactor"] in C.SIGMA_FACTORS, name
        assert 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_default_is_a_valid_widget_value():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_the_presets_show_the_documented_outcomes():
    res = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    std, gcv, lc, sg = (res[k] for k in C.PRESETS)
    assert not std.failed and std.err_rule < 0.13
    assert gcv.failed and gcv.lam_rule / gcv.lam_star < 0.01 and 0.8 < gcv.err_rule < 0.9
    assert lc.failed and 0.04 < lc.lam_rule / lc.lam_star < 0.09 and 0.3 < lc.err_rule < 0.35
    assert sg.failed and sg.lam_rule / sg.lam_star < 0.05 and 0.8 < sg.err_rule < 0.9


def test_experiment_constants_are_consistent():
    assert C.DEFAULT_SFACTOR == 1.0 and 0.25 in C.SIGMA_FACTORS and 4.0 in C.SIGMA_FACTORS and C.EXP_REPS >= 200
    assert C.DEFAULT_NOISE in C.EXP_NOISE_LEVELS and C.DEFAULT_N in C.EXP_N_SWEEP and set(C.EXP_OPERATORS) <= set(C.OPERATOR_LABELS)
    assert 1.0 in C.GCV_BOUND_FACTORS and 10.0 in C.GCV_BOUND_FACTORS and C.FAIL_FACTOR == 2.0
