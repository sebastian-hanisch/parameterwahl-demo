"""Auswertung: Kennzahlen der Instanz, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import prw_constants as C
import prw_model as model
from inverse_generator import SplitMix64, blurred_exact, grid, noise_sigma, operator, phantom

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def lam_text(lam: float) -> str:
    """Strafparameter: ab 0,01 mit Dezimalstellen, darunter als Zehnerpotenz."""
    if lam == 0.0:
        return "0"
    return de(lam, 3 if lam >= 0.1 else 4) if lam >= 0.01 else sci(lam)


def err_pct(err: float) -> str:
    """Fehler als Prozent der Signalnorm: ab 100 Prozent ohne, ab 1 Prozent mit einer, ab 0,01 Prozent mit zwei Stellen, darunter als Zehnerpotenz;
    ab dem Zehnfachen der Signalnorm als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich"
    if err >= 10.0:
        return f"{sci(err)}-fach"
    x = 100.0 * err
    if x >= 99.5:
        return f"{de(x, 0)} %"
    if x >= 0.95:
        return f"{de(x, 1)} %"
    if x >= 0.01:
        return f"{de(x, 2)} %"
    return f"{sci(x)} %"


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


def ratio_text(r: float) -> str:
    """Verhältnis zweier Parameter: ab 0,01 mit zwei Stellen, darunter als Zehnerpotenz."""
    return de(r, 2) if r >= 0.01 else sci(r)


def share(x: float) -> str:
    """Anteil (0 bis 1) als ganze Prozent."""
    return f"{de(100.0 * x, 0)} %"


# ----------------------------------------------------------------------------- Welt

@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    w: float = C.DEFAULT_W
    noise: float = C.DEFAULT_NOISE
    data: str = "ehrlich"
    rule: str = "dp"
    sfactor: float = C.DEFAULT_SFACTOR
    seed: int = C.DEFAULT_SEED


class World:
    """Matrix, Zerlegung, Wahrheit, Modellfehler und alle Auswertungen über das λ-Gitter."""

    def __init__(self, n: int, w: float, noise: float | None = None, sigma: float | None = None, a: np.ndarray | None = None):
        self.n, self.w = n, w
        self.a = operator(n, w) if a is None else a
        self.u, self.sv, self.vt = model.decompose(self.a)
        self.x = phantom(grid(n))
        self.g = blurred_exact(grid(n), w) if a is None else self.a @ self.x
        self.c = self.vt @ self.x
        self.rho_honest = (self.u.T @ (self.g - self.a @ self.x)) / np.maximum(self.sv, 1e-300)
        if sigma is not None:
            self.sigma = float(sigma)
        elif a is None:
            self.sigma = noise_sigma(n, w, noise)
        else:
            self.sigma = float(noise) * float(np.sqrt(np.mean(self.g ** 2)))
        self.x_norm = float(np.linalg.norm(self.x))
        self.lams = model.lambda_grid()
        self.cond = float(self.sv[0] / self.sv[-1]) if self.sv[-1] > 0 else math.inf

    def honest_flag_base(self, honest: bool) -> np.ndarray:
        return self.g if honest else self.a @ self.x

    def data(self, honest: bool, reps: int, seed: int) -> np.ndarray:
        e = SplitMix64(seed).normal(self.n * reps).reshape(reps, self.n) * self.sigma
        return self.honest_flag_base(honest)[None, :] + e

    def expected_best(self, honest: bool) -> tuple[int, float, float]:
        """(Index, λ*, erwarteter Fehler) des Optimums über das Gitter, Wahrheit bekannt (Orakel)."""
        rho = self.rho_honest if honest else np.zeros(self.n)
        e = model.expected_errors(self.sv, self.c, rho, self.sigma, self.lams, self.x_norm)
        i = int(np.argmin(e))
        return i, float(self.lams[i]), float(e[i])

    def evaluate(self, b: np.ndarray, sigma_assumed=None, tau: float = C.TAU) -> dict:
        """Alle Regeln und das Realisierungsorakel für Daten b der Form (reps, n): Indizes und tatsächliche Fehler über das Gitter."""
        cv = model.curves(self.u, self.sv, b, self.lams)
        err = model.errors(self.vt, self.sv, cv["beta"], self.lams, self.c, self.x_norm)
        sg = self.sigma if sigma_assumed is None else sigma_assumed
        idx = {
            "dp": model.discrepancy_index(cv["res2"], sg, self.n, tau),
            "gcv": model.gcv_index(cv["res2"], cv["df"], self.n),
            "lcurve": model.lcurve_index(cv["res2"], cv["sol2"], self.lams),
            "oracle": model.oracle_index(err),
        }
        return {"cv": cv, "err": err, "idx": idx}

    def chosen_errors(self, ev: dict, key: str) -> np.ndarray:
        return ev["err"][np.arange(ev["err"].shape[0]), ev["idx"][key]]


def summarize(errs: np.ndarray, lam_ratio: np.ndarray, err_star: float) -> dict:
    """Kennzahlen über die Wiederholungen: RMS, Median, 90. Perzentil, Anteil über dem Doppelten des erwarteten Optimums, Median von λ/λ*."""
    return {"rms": float(np.sqrt(np.mean(errs ** 2))), "median": float(np.median(errs)), "p90": float(np.quantile(errs, 0.9)),
            "fail": float(np.mean(errs > C.FAIL_FACTOR * err_star)), "lam_ratio": float(np.median(lam_ratio))}


# ----------------------------------------------------------------------------- Instanz

@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray
    x: np.ndarray
    g: np.ndarray
    sv: np.ndarray
    cond: float
    sigma: float
    lams: np.ndarray
    b: np.ndarray
    x_naive: np.ndarray
    x_rule: np.ndarray
    x_oracle: np.ndarray
    lam_rule: float
    lam_oracle: float
    lam_star: float
    err_star: float
    err_rule: float
    err_oracle: float
    err_naive: float
    failed: bool
    picks: dict              # Index je Regel
    pick_errors: dict        # Fehler dieser Stichprobe je Regel
    err_curve: np.ndarray    # tatsächlicher Fehler über das Gitter
    exp_curve: np.ndarray    # erwarteter Fehler über das Gitter
    res: np.ndarray          # ||A x_lam - b||
    sol: np.ndarray          # ||x_lam||
    gcv: np.ndarray
    kappa: np.ndarray
    target: float            # Ziel des Diskrepanzprinzips: tau * sigma_angenommen * sqrt(n)
    x_norm: float
    singular: bool


def analyse(settings: Settings) -> Analysis:
    st = settings
    wd = World(st.n, st.w, st.noise)
    honest = st.data != "verbrechen"
    b = wd.data(honest, 1, st.seed)
    ev = wd.evaluate(b, sigma_assumed=wd.sigma * st.sfactor)
    cv, err, idx = ev["cv"], ev["err"][0], {k: int(v[0]) for k, v in ev["idx"].items()}
    _, lam_star, err_star = wd.expected_best(honest)
    f = model.filter_matrix(wd.sv, wd.lams)
    beta = cv["beta"][0]

    def sol_at(i: int) -> np.ndarray:
        return wd.vt.T @ ((f[i] / wd.sv) * beta)

    pick = idx[st.rule]
    x_naive = wd.vt.T @ (beta / wd.sv)
    rho = wd.rho_honest if honest else np.zeros(st.n)
    exp_curve = model.expected_errors(wd.sv, wd.c, rho, wd.sigma, wd.lams, wd.x_norm)
    gcv = model.gcv_curve(cv["res2"], cv["df"], st.n)[0]
    return Analysis(
        settings=st, s=grid(st.n), x=wd.x, g=wd.g, sv=wd.sv, cond=wd.cond, sigma=wd.sigma, lams=wd.lams, b=b[0], x_naive=x_naive, x_rule=sol_at(pick),
        x_oracle=sol_at(idx["oracle"]), lam_rule=float(wd.lams[pick]), lam_oracle=float(wd.lams[idx["oracle"]]), lam_star=lam_star, err_star=err_star,
        err_rule=float(err[pick]), err_oracle=float(err[idx["oracle"]]), err_naive=float(np.sqrt(np.sum((x_naive - wd.x) ** 2)) / wd.x_norm),
        failed=bool(err[pick] > C.FAIL_FACTOR * err_star), picks=idx, pick_errors={k: float(err[v]) for k, v in idx.items()}, err_curve=err,
        exp_curve=exp_curve, res=np.sqrt(cv["res2"][0]), sol=np.sqrt(cv["sol2"][0]), gcv=gcv,
        kappa=model.lcurve_curvature(cv["res2"], cv["sol2"], wd.lams)[0], target=C.TAU * wd.sigma * st.sfactor * math.sqrt(st.n),
        x_norm=wd.x_norm, singular=bool(wd.cond >= C.SINGULAR_COND),
    )


def standard_sigma() -> float:
    """Absolutes Rauschen der Standardinstanz: bleibt in Experiment 2 fest, wenn das Gitter wechselt."""
    return noise_sigma(C.DEFAULT_N, C.DEFAULT_W, C.DEFAULT_NOISE)


# ----------------------------------------------------------------------------- Experiment 1: wer trifft?

def who_experiment(reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE) -> dict:
    """Die drei Regeln und das Realisierungsorakel über feste Ströme, ehrliche Daten und Daten aus demselben Gitter."""
    wd = World(n, w, noise)
    out = {"n": n, "reps": reps, "kinds": {}}
    for kind, honest in (("honest", True), ("crime", False)):
        _, lam_star, err_star = wd.expected_best(honest)
        ev = wd.evaluate(wd.data(honest, reps, C.SEED_BASE + (1 if honest else 2)))
        rules = {}
        for key in ("dp", "gcv", "lcurve", "oracle"):
            e = wd.chosen_errors(ev, key)
            ratio = wd.lams[ev["idx"][key]] / lam_star
            rules[key] = {**summarize(e, ratio, err_star), "errors": e, "ratio_q": np.quantile(ratio, [0.05, 0.25, 0.5, 0.75, 0.95])}
        # Anatomie der GCV-Katastrophen: hintere Komponenten und direkte Inversion der gescheiterten Wiederholungen
        beta = ev["cv"]["beta"]
        tail = model.noise_tail_estimate(beta) / wd.sigma
        naive = np.sqrt((((beta / wd.sv) @ wd.vt - wd.x) ** 2).sum(axis=1)) / wd.x_norm
        failed = rules["gcv"]["errors"] > C.FAIL_FACTOR * err_star
        anatomy = {"tail_failed": float(np.median(tail[failed])) if failed.any() else float("nan"), "tail_ok": float(np.median(tail[~failed])),
                   "naive_failed": float(np.median(naive[failed])) if failed.any() else float("nan"), "naive_all": float(np.median(naive)),
                   "tiny_lambda": float(np.mean(wd.lams[ev["idx"]["gcv"]][failed] / lam_star < 0.01)) if failed.any() else float("nan")}
        out["kinds"][kind] = {"lam_star": lam_star, "err_star": err_star, "rules": rules, "gcv_anatomy": anatomy}
    return out


# ----------------------------------------------------------------------------- Experiment 2: Rauschstufen und Gitter

def noise_experiment(levels=C.EXP_NOISE_LEVELS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, reps: int = C.EXP_REPS) -> list[dict]:
    rows = []
    for i, nz in enumerate(levels):
        wd = World(n, w, nz)
        row = {"noise": nz}
        for kind, honest in (("honest", True), ("crime", False)):
            _, lam_star, err_star = wd.expected_best(honest)
            ev = wd.evaluate(wd.data(honest, reps, C.SEED_BASE + 10 + i))
            row[kind] = {"lam_star": lam_star, "err_star": err_star,
                         **{k: summarize(wd.chosen_errors(ev, k), wd.lams[ev["idx"][k]] / lam_star, err_star) for k in ("dp", "gcv", "lcurve", "oracle")}}
        rows.append(row)
    return rows


def grid_experiment(ns=C.EXP_N_SWEEP, w: float = C.DEFAULT_W, reps: int = C.EXP_REPS) -> list[dict]:
    """Gleiches absolutes Rauschen der Standardinstanz, ehrliche Daten, je Gitter."""
    sg = standard_sigma()
    rows = []
    for i, n in enumerate(ns):
        wd = World(n, w, sigma=sg)
        _, lam_star, err_star = wd.expected_best(True)
        ev = wd.evaluate(wd.data(True, reps, C.SEED_BASE + 30 + i))
        rows.append({"n": n, "lam_star": lam_star, "err_star": err_star,
                     **{k: summarize(wd.chosen_errors(ev, k), wd.lams[ev["idx"][k]] / lam_star, err_star) for k in ("dp", "gcv", "lcurve", "oracle")}})
    return rows


# ----------------------------------------------------------------------------- Experiment 3: das Diskrepanzprinzip braucht σ

def sigma_experiment(reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE) -> dict:
    wd = World(n, w, noise)
    _, lam_star, err_star = wd.expected_best(True)
    b = wd.data(True, reps, C.SEED_BASE + 40)
    rows = []
    for f in C.SIGMA_FACTORS:
        ev = wd.evaluate(b, sigma_assumed=wd.sigma * f)
        e = wd.chosen_errors(ev, "dp")
        rows.append({"factor": f, **summarize(e, wd.lams[ev["idx"]["dp"]] / lam_star, err_star)})
    taus = []
    for t in (1.0, 1.1, 1.5, 2.0):
        ev = wd.evaluate(b, tau=t)
        e = wd.chosen_errors(ev, "dp")
        taus.append({"tau": t, **summarize(e, wd.lams[ev["idx"]["dp"]] / lam_star, err_star)})
    tail = []
    for i, nz in enumerate(C.EXP_NOISE_LEVELS):
        for kind, honest in (("honest", True), ("crime", False)):
            w2 = World(n, w, nz)
            _, ls, es = w2.expected_best(honest)
            bb = w2.data(honest, reps, C.SEED_BASE + 50 + i)
            ev = w2.evaluate(bb)
            sh = model.noise_tail_estimate(ev["cv"]["beta"])
            idx_hat = model.discrepancy_index(ev["cv"]["res2"], sh, n)
            e_hat = ev["err"][np.arange(reps), idx_hat]
            tail.append({"noise": nz, "kind": kind, "ratio": float(np.median(sh) / w2.sigma), "dp_true": summarize(w2.chosen_errors(ev, "dp"), w2.lams[ev["idx"]["dp"]] / ls, es),
                         "dp_hat": summarize(e_hat, w2.lams[idx_hat] / ls, es), "oracle": summarize(w2.chosen_errors(ev, "oracle"), w2.lams[ev["idx"]["oracle"]] / ls, es)})
    return {"rows": rows, "taus": taus, "tail": tail, "lam_star": lam_star, "err_star": err_star, "n": n, "reps": reps}


# ----------------------------------------------------------------------------- Experiment 4: andere Matrizen und die Untergrenze für GCV

def other_experiment(reps: int = C.EXP_REPS, n: int = C.DEFAULT_N) -> dict:
    ops = []
    for i, op in enumerate(C.EXP_OPERATORS):
        a = model.integration_matrix(n, 1 if op == "integral" else 2)
        for j, nz in enumerate(C.EXP_OP_NOISE):
            wd = World(n, 0.0, noise=nz, a=a)
            _, lam_star, err_star = wd.expected_best(False)
            ev = wd.evaluate(wd.data(False, reps, C.SEED_BASE + 60 + 10 * i + j))
            ops.append({"op": op, "noise": nz, "lam_star": lam_star, "err_star": err_star,
                        **{k: summarize(wd.chosen_errors(ev, k), wd.lams[ev["idx"][k]] / lam_star, err_star) for k in ("dp", "gcv", "lcurve", "oracle")}})
    bounded = []
    for kind, honest in (("honest", True), ("crime", False)):
        wd = World(n, C.DEFAULT_W, C.DEFAULT_NOISE)
        _, lam_star, err_star = wd.expected_best(honest)
        ev = wd.evaluate(wd.data(honest, reps, C.SEED_BASE + 80 + (0 if honest else 1)))
        base = summarize(wd.chosen_errors(ev, "gcv"), wd.lams[ev["idx"]["gcv"]] / lam_star, err_star)
        rows = [{"factor": 0.0, **base}]
        for fac in C.GCV_BOUND_FACTORS:
            ix = model.gcv_index_bounded(ev["cv"]["res2"], ev["cv"]["df"], n, wd.lams, fac * float(wd.sv[-1]))
            e = ev["err"][np.arange(reps), ix]
            rows.append({"factor": fac, **summarize(e, wd.lams[ix] / lam_star, err_star)})
        bounded.append({"kind": kind, "rows": rows, "err_star": err_star})
    return {"ops": ops, "bounded": bounded, "n": n, "reps": reps}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    rule = C.RULE_SHORT[a.settings.rule]
    hint = f"Das Orakel dieser Stichprobe wählt λ = {lam_text(a.lam_oracle)} mit {err_pct(a.err_oracle)}; der beste erwartete Parameter ist λ* = {lam_text(a.lam_star)} mit {err_pct(a.err_star)}."
    if a.failed:
        side = "zu klein" if a.lam_rule < a.lam_star else "zu groß"
        return "warning", (f"{rule} wählt λ = {lam_text(a.lam_rule)} ({ratio_text(a.lam_rule / a.lam_star)}-mal λ*, {side}): Fehler {err_pct(a.err_rule)}, mehr als das Doppelte "
                           f"des erwarteten Optimums. {hint}")
    return "success", f"{rule} wählt λ = {lam_text(a.lam_rule)} ({ratio_text(a.lam_rule / a.lam_star)}-mal λ*) mit Fehler {err_pct(a.err_rule)}. {hint}"


def verdict_who(res: dict) -> tuple[str, str]:
    h = res["kinds"]["honest"]["rules"]
    return "warning", (f"Ehrliche Daten, {res['reps']} feste Wiederholungen: Das Diskrepanzprinzip liegt im Median bei {err_pct(h['dp']['median'])} ({share(h['dp']['fail'])} über dem "
                       f"Doppelten des Optimums), GCV bei {err_pct(h['gcv']['median'])} mit {share(h['gcv']['fail'])} Katastrophen (90. Perzentil {err_pct(h['gcv']['p90'])}), die "
                       f"L-Kurve bei {err_pct(h['lcurve']['median'])} mit {share(h['lcurve']['fail'])}. Das Realisierungsorakel erreicht {err_pct(h['oracle']['median'])}.")


def verdict_noise(rows: list[dict]) -> tuple[str, str]:
    mid = next(r for r in rows if abs(r["noise"] - C.DEFAULT_NOISE) < 1e-12)
    last = rows[-1]
    return "info", (f"Bei {noise_label(mid['noise'])} Rauschen scheitert GCV in {share(mid['honest']['gcv']['fail'])} der Wiederholungen (ehrliche Daten), die L-Kurve in "
                    f"{share(mid['honest']['lcurve']['fail'])}, das Diskrepanzprinzip in {share(mid['honest']['dp']['fail'])}. Bei {noise_label(last['noise'])} und Daten aus "
                    f"demselben Gitter liegt das Diskrepanzprinzip bei {err_pct(last['crime']['dp']['median'])} statt {err_pct(last['crime']['oracle']['median'])}.")


def verdict_sigma(res: dict) -> tuple[str, str]:
    by = {r["factor"]: r for r in res["rows"]}
    return "warning", (f"Mit einem um den Faktor 4 zu kleinen σ liegt der Fehler bei {err_pct(by[0.25]['rms'])}, bei Faktor 2 zu klein {err_pct(by[0.5]['rms'])}; mit "
                       f"vierfach zu großem σ nur bei {err_pct(by[4.0]['rms'])} (richtig: {err_pct(by[1.0]['rms'])}). Unterschätzen ist gefährlich, Überschätzen billig.")


def verdict_other(res: dict) -> tuple[str, str]:
    worst = max(res["ops"], key=lambda r: r["gcv"]["fail"])
    return "info", (f"Auch auf anderen Matrizen ist das Diskrepanzprinzip nie schlechter als {de(100.0 * (max(r['dp']['rms'] / r['err_star'] for r in res['ops']) - 1.0), 0)} % über "
                    f"dem Optimum; GCV scheitert dort bis zu {share(worst['gcv']['fail'])} der Wiederholungen ({C.OPERATOR_LABELS[worst['op']]}, {noise_label(worst['noise'])}).")
