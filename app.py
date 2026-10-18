"""Parameterwahl ohne Wahrheit: GCV, Diskrepanzprinzip und L-Kurve - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Linie Inverse Probleme und Regularisierung. Auf der Welt aus Stück 1 bis 3 (Phantom, Gauß-Unschärfe, Rauschen) wählen drei Regeln den
Tikhonov-Parameter aus den Daten allein; das Orakel (Wahrheit bekannt) steht als Maßstab daneben. Kern: prw_model.py, Generator: inverse_generator.py
(wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import prw_constants as C
from prw_evaluation import (
    Settings,
    analyse,
    de,
    err_pct,
    grid_experiment,
    lam_text,
    noise_experiment,
    noise_label,
    other_experiment,
    ratio_text,
    share,
    sci,
    sigma_experiment,
    standard_sigma,
    verdict_instance,
    verdict_noise,
    verdict_other,
    verdict_sigma,
    verdict_who,
    who_experiment,
)
from prw_pdf_export import build_pdf
from prw_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from prw_visualization import (
    build_box_chart,
    build_error_chart,
    build_fail_chart,
    build_gcv_chart,
    build_lcurve_chart,
    build_other_chart,
    build_recon_chart,
    build_residual_chart,
    build_sigma_chart,
    build_table,
    style_table,
)

st.set_page_config(page_title="Parameterwahl – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _who():
    return who_experiment()


@st.cache_data(show_spinner=SPINNER)
def _noise():
    return noise_experiment()


@st.cache_data(show_spinner=SPINNER)
def _grid():
    return grid_experiment()


@st.cache_data(show_spinner=SPINNER)
def _sigma():
    return sigma_experiment()


@st.cache_data(show_spinner=SPINNER)
def _other():
    return other_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🧭 Parameterwahl ohne Wahrheit: GCV, Diskrepanzprinzip, L-Kurve")
st.markdown(
    """
**Das beste λ kennt nur, wer die Wahrheit kennt.** Dies ist das vierte Stück der Linie **Inverse Probleme und Regularisierung** und baut auf **Tikhonov-Regularisierung**
und **Abgeschnittene SVD** auf: dasselbe Phantom, dieselbe Gauß-Unschärfe, dasselbe Rauschen. Drei Regeln wählen λ aus den Daten allein: das **Diskrepanzprinzip** (braucht
das Rauschen σ), die **generalisierte Kreuzvalidierung (GCV)** und die **L-Kurve**. Die Welt ist simuliert; das Orakel dieser Stichprobe und das beste erwartete λ\\* stehen
als Maßstab daneben.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So wählt man λ ohne die Wahrheit", expanded=True):
    st.markdown(
        r"""
1. **Diskrepanzprinzip.** Die Lösung soll die Daten nur so genau anpassen, wie das Rauschen es erlaubt: das größte $\lambda$ mit $\|A x_\lambda - b\|^2 \le \tau^2\sigma^2 n$.
   Es braucht $\sigma$.
2. **GCV.** Wählt das $\lambda$, das ein Weglassen einzelner Messwerte am besten vorhersagt: Minimum von $\|A x_\lambda - b\|^2/(n - \mathrm{df})^2$. Es braucht kein $\sigma$.
3. **L-Kurve.** Trägt $\log\|x_\lambda\|$ gegen $\log\|A x_\lambda - b\|$ auf und nimmt die Ecke der größten Krümmung. Es braucht kein $\sigma$.
4. **Maßstab.** Das *Orakel dieser Stichprobe* wählt das λ mit dem kleinsten tatsächlichen Fehler; das beste erwartete $\lambda^*$ minimiert den Fehler im Mittel über das Rauschen.
   Beides braucht die Wahrheit und steht nur zum Vergleich da.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Messung**")
    n = st.slider("Gitterpunkte n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Wie viele Messpunkte und Unbekannte.")
    w = st.slider("Breite der Unschärfe w", *bounds("w_slider"), step=C.W_STEP, key="w_slider", format="%.3f",
                  help="Breite des Gauß-Kerns als Anteil der Länge 1. Je breiter, desto stärker fallen die Singulärwerte.")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label,
                             help="Standardabweichung des weißen Rauschens, relativ zum Effektivwert der exakten Daten.")
    st.markdown("**Daten**")
    data = st.selectbox("Woher kommen die Daten?", C.DATA_CHOICES, key="data_select", format_func=lambda v: C.DATA_LABELS[v],
                        help="Ehrliche Daten kommen aus der analytischen Unschärfe und sind vom Gitter unabhängig. Daten aus demselben Gitter entstehen aus "
                             "der Matrix, die danach invertiert wird (inverse crime).")
    st.markdown("**Regel**")
    rule = st.selectbox("Welche Regel wählt λ?", C.RULES, key="rule_select", format_func=lambda v: C.RULE_LABELS[v],
                        help="Die gewählte Regel bestimmt die Rekonstruktion in der Primäransicht; die Fehler aller drei Regeln stehen im Fehlerdiagramm.")
    sfactor = st.select_slider("Angenommenes σ (Vielfaches des wahren)", options=C.SIGMA_FACTORS, key="sfactor_select", format_func=lambda v: f"{de(v, 2)}×",
                               help="Wirkt nur beim Diskrepanzprinzip: Mit 1 kennt die Regel das wahre Rauschen, mit 0,25 unterschätzt sie es um den Faktor 4.")
    seed = st.number_input("Zufalls-Seed des Rauschens", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Rauschen würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "w_slider": round(float(w), 3), "noise_select": float(noise), "data_select": data, "rule_select": rule,
                   "sfactor_select": float(sfactor), "seed_input": int(seed)})

settings = Settings(int(n), round(float(w), 3), float(noise), data, rule, float(sfactor), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Gewähltes λ ({C.RULE_SHORT[rule]})", lam_text(a.lam_rule), delta=f"{ratio_text(a.lam_rule / a.lam_star)}-mal λ*", delta_color="off",
          help="Das λ, das die gewählte Regel aus den Daten dieser Stichprobe bestimmt.")
c2.metric("Fehler dieser Stichprobe", err_pct(a.err_rule), delta=f"Orakel dieser Stichprobe: {err_pct(a.err_oracle)}", delta_color="off",
          help="‖x̂ − x‖ / ‖x‖ mit der Wahrheit x des Phantoms; das Orakel wählt das λ mit dem kleinsten tatsächlichen Fehler dieser Stichprobe.")
c3.metric("Bestes erwartetes λ*", lam_text(a.lam_star), delta=f"erwarteter Fehler {err_pct(a.err_star)}", delta_color="off",
          help="Minimum des erwarteten Fehlers über das Rauschen (geschlossen); braucht die Wahrheit.")
c4.metric("Direkte Inversion", err_pct(a.err_naive), delta="λ = 0 (Stück 1)", delta_color="off", help="Fehler dieser Stichprobe ohne Strafe.")

tab_rec, tab_err, tab_gcv, tab_dp, tab_lc, tab_tab = st.tabs(["Rekonstruktion", "Fehler über λ", "GCV", "Diskrepanzprinzip", "L-Kurve", "Tabelle"])
with tab_rec:
    st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
    st.caption("Grün die Wahrheit, grau gestrichelt das Orakel dieser Stichprobe, farbig die gewählte Regel, blass rot die direkte Inversion. Die Achse ist fest.")
with tab_err:
    st.plotly_chart(build_error_chart(a), width="stretch", key="error_chart")
    st.caption("Der tatsächliche Fehler dieser Stichprobe über λ (schwarz) und der erwartete (gepunktet). Die Marken zeigen, wo jede Regel landet; das Orakel sitzt im Minimum.")
with tab_gcv:
    st.plotly_chart(build_gcv_chart(a), width="stretch", key="gcv_chart")
    st.caption("Die GCV-Funktion dieser Stichprobe. Hat sie ein zweites, tieferes Minimum bei sehr kleinem λ, wählt GCV es: das Rauschen kommt dann ungedämpft durch.")
with tab_dp:
    st.plotly_chart(build_residual_chart(a), width="stretch", key="residual_chart")
    st.caption("Das Residuum wächst mit λ. Das Diskrepanzprinzip nimmt das größte λ, dessen Residuum noch unter dem Ziel liegt; ein zu kleines angenommenes σ senkt das Ziel.")
with tab_lc:
    st.plotly_chart(build_lcurve_chart(a), width="stretch", key="lcurve_chart")
    st.caption("Die L-Kurve in Doppellogarithmus-Darstellung. Die Ecke der größten Krümmung liegt hier links von λ*: zu wenig Glättung.")
with tab_tab:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=min(40 + 35 * min(settings.n, 40), 560))
    st.caption("Die ersten Punkte der Instanz.")

if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="parameterwahl.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist dieser Stichprobe zu trauen?")
r1, r2, r3 = st.columns(3)
for col, key in zip((r1, r2, r3), ("dp", "gcv", "lcurve")):
    col.metric(f"{C.RULE_SHORT[key]}: Fehler dieser Stichprobe", err_pct(a.pick_errors[key]), delta=f"λ = {lam_text(a.lams[a.picks[key]])}", delta_color="off")
verdict(verdict_instance(a))
st.caption("Eine einzelne Stichprobe zeigt nicht, wie oft eine Regel scheitert: dafür sind die Experimente mit festen Wiederholungen da.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wer trifft?")
res1 = _who()
st.caption(f"Standardinstanz (n = {res1['n']}), {res1['reps']} feste Wiederholungen je Datenart. Gezeigt ist die Verteilung des Fehlers über die Wiederholungen je Regel; "
           "gescheitert heißt: mehr als das Doppelte des erwarteten Optimums.")
kind = st.radio("Datenart", ("honest", "crime"), key="who_kind", horizontal=True, format_func=lambda v: "ehrliche Daten" if v == "honest" else "Daten aus demselben Gitter")
st.plotly_chart(build_box_chart(res1, kind), width="stretch", key="box_chart")
verdict(verdict_who(res1))
d1 = res1["kinds"][kind]
st.dataframe(
    [{"Regel": C.RULE_SHORT[k], "Median": err_pct(v["median"]), "90. Perzentil": err_pct(v["p90"]), "RMS": err_pct(v["rms"]), "Gescheitert": share(v["fail"]),
      "Median λ/λ*": de(v["lam_ratio"], 2), "5 % / 95 % von λ/λ*": f"{de(v['ratio_q'][0], 2)} / {de(v['ratio_q'][4], 2)}"} for k, v in d1["rules"].items()],
    hide_index=True, width="stretch")
st.caption(f"λ* = {lam_text(d1['lam_star'])}, erwarteter Fehler {err_pct(d1['err_star'])}. Der RMS-Mittelwert wird bei GCV von einzelnen Katastrophen bestimmt, deshalb stehen "
           "Median, 90. Perzentil und der Anteil Gescheiterter daneben. Das Diskrepanzprinzip wählt λ im Median doppelt so groß wie λ*, und der Fehler bleibt flach.")
an = d1["gcv_anatomy"]
st.caption(f"**Warum GCV scheitert:** In den gescheiterten Wiederholungen sind die hinteren Komponenten zufällig klein (Median {de(an['tail_failed'], 2)}·σ gegen "
           f"{de(an['tail_ok'], 2)}·σ sonst); die Daten sehen fast rauschfrei aus, und GCV wählt λ gegen null ({share(an['tiny_lambda'])} der Fälle unter 1 % von λ*). Der Fehler "
           f"entspricht dann der direkten Inversion dieser Stichprobe (Median {err_pct(an['naive_failed'])} gegen {err_pct(an['naive_all'])} über alle Wiederholungen).")

st.subheader("🔬 Experiment 2: Rauschstufen und Gitter")
rows2 = _noise()
st.caption(f"Standardinstanz (n = {C.DEFAULT_N}), {C.EXP_REPS} feste Wiederholungen je Rauschstufe; Anteil der Wiederholungen mit mehr als dem Doppelten des erwarteten Optimums.")
kind2 = st.radio("Datenart", ("honest", "crime"), key="noise_kind", horizontal=True, format_func=lambda v: "ehrliche Daten" if v == "honest" else "Daten aus demselben Gitter")
st.plotly_chart(build_fail_chart(rows2, kind2), width="stretch", key="fail_chart")
verdict(verdict_noise(rows2))
st.dataframe(
    [{"Rauschen": noise_label(r["noise"]), "λ*": lam_text(r[kind2]["lam_star"]), "Optimum": err_pct(r[kind2]["err_star"]),
      "Diskrepanz (Median)": err_pct(r[kind2]["dp"]["median"]), "GCV (Median)": err_pct(r[kind2]["gcv"]["median"]), "L-Kurve (Median)": err_pct(r[kind2]["lcurve"]["median"]),
      "Orakel (Median)": err_pct(r[kind2]["oracle"]["median"])} for r in rows2],
    hide_index=True, width="stretch")
rows2b = _grid()
st.markdown("**Gitter** (gleiches absolutes Rauschen der Standardinstanz %s, ehrliche Daten)" % sci(standard_sigma()))
st.dataframe(
    [{"n": r["n"], "λ*": lam_text(r["lam_star"]), "Optimum": err_pct(r["err_star"]), "Diskrepanz (Median)": err_pct(r["dp"]["median"]), "GCV (Median)": err_pct(r["gcv"]["median"]),
      "GCV RMS": err_pct(r["gcv"]["rms"]), "GCV gescheitert": share(r["gcv"]["fail"]), "L-Kurve (Median)": err_pct(r["lcurve"]["median"]),
      "L-Kurve gescheitert": share(r["lcurve"]["fail"])} for r in rows2b],
    hide_index=True, width="stretch")
st.caption("Bei n = 24 und 32 braucht es kaum eine Strafe (λ* an der Untergrenze): alle Regeln gleich dem Optimum. Ab n = 48 trennen sich die Regeln.")

st.subheader("🔬 Experiment 3: Das Diskrepanzprinzip braucht σ")
res3 = _sigma()
st.caption(f"Standardinstanz (n = {res3['n']}), ehrliche Daten, {res3['reps']} feste Wiederholungen: das Diskrepanzprinzip mit einem falsch angenommenen σ.")
st.plotly_chart(build_sigma_chart(res3), width="stretch", key="sigma_chart")
verdict(verdict_sigma(res3))
st.dataframe(
    [{"σ angenommen / wahr": de(r["factor"], 2), "RMS-Fehler": err_pct(r["rms"]), "Median": err_pct(r["median"]), "Median λ/λ*": de(r["lam_ratio"], 2)} for r in res3["rows"]],
    hide_index=True, width="stretch")
st.markdown("**Rauschschätzung aus den letzten %d Komponenten** (σ̂ = Effektivwert von |uᵢᵀb|, dann Diskrepanzprinzip mit σ̂)" % C.TAIL_M)
st.dataframe(
    [{"Rauschen": noise_label(r["noise"]), "Daten": "ehrlich" if r["kind"] == "honest" else "selbes Gitter", "σ̂/σ (Median)": de(r["ratio"], 2),
      "Diskrepanz mit σ (RMS)": err_pct(r["dp_true"]["rms"]), "Diskrepanz mit σ̂ (RMS)": err_pct(r["dp_hat"]["rms"]), "Orakel (RMS)": err_pct(r["oracle"]["rms"])} for r in res3["tail"]],
    hide_index=True, width="stretch", height=420)
st.caption("Die Schätzung trägt, solange die hinteren Komponenten nur Rauschen enthalten. Bei sehr kleinem Rauschen stecken dort noch Signal und Modellfehler, und σ̂ liegt weit "
           "zu hoch; ebenso schwankt σ̂ bei nur acht Werten um rund ein Viertel.")

st.subheader("🔬 Experiment 4: Andere Matrizen und die Untergrenze für GCV")
res4 = _other()
st.caption(f"Gleiches Phantom, zwei Matrizen mit algebraisch fallenden Singulärwerten (n = {res4['n']}), {res4['reps']} feste Wiederholungen; darunter GCV mit Untergrenze "
           "λ ≥ Faktor · σₙ (kleinster Singulärwert) auf der Standardinstanz.")
st.plotly_chart(build_other_chart(res4), width="stretch", key="other_chart")
verdict(verdict_other(res4))
st.dataframe(
    [{"Matrix": C.OPERATOR_LABELS[r["op"]], "Rauschen": noise_label(r["noise"]), "Optimum": err_pct(r["err_star"]), "Diskrepanz (RMS)": err_pct(r["dp"]["rms"]),
      "GCV (RMS)": err_pct(r["gcv"]["rms"]), "GCV gescheitert": share(r["gcv"]["fail"]), "L-Kurve (RMS)": err_pct(r["lcurve"]["rms"]),
      "L-Kurve gescheitert": share(r["lcurve"]["fail"])} for r in res4["ops"]],
    hide_index=True, width="stretch")
st.dataframe(
    [{"Daten": "ehrlich" if b["kind"] == "honest" else "selbes Gitter", "Untergrenze": "keine" if r["factor"] == 0.0 else f"λ ≥ {de(r['factor'], 0)}·σₙ", "Median": err_pct(r["median"]),
      "90. Perzentil": err_pct(r["p90"]), "Gescheitert": share(r["fail"])} for b in res4["bounded"] for r in b["rows"]],
    hide_index=True, width="stretch")
st.caption("Eine Untergrenze gegen Fehlminima am Gitterrand beseitigt die Katastrophen nicht: Die Fehlminima liegen nicht nur am Rand.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **σ ist bekannt (Diskrepanzprinzip)** | Ein um den Faktor 4 zu kleines σ liefert {err_pct(res3['rows'][0]['rms'])} statt {err_pct(res3['rows'][3]['rms'])}; Überschätzen kostet kaum etwas. | Bayes-Sicht (σ als Teil des Modells) |
| **Das Rauschen ist weiß und Gauß-verteilt** | Alle drei Regeln nehmen weißes Rauschen an; korreliertes Rauschen verschiebt jede von ihnen. | – |
| **Ein Phantom, eine Matrix, nur Tikhonov** | Drei Matrizen, ein Phantom; die Rangwahl der abgeschnittenen SVD mit denselben Regeln ist nicht gemessen. | – |
| **„Gescheitert“ ist ein Schwellenwert** | Mehr als das Doppelte des erwarteten Optimums: eine Konvention, keine Naturgröße; der Anteil ändert sich mit ihr. | – |
| **Das Orakel kennt die Wahrheit** | λ* und das Realisierungsorakel sind in der Praxis nicht verfügbar; sie sind nur der Maßstab. | – |
| **L-Kurve per endlicher Differenzen** | Die Krümmung ist auf dem Gitter ohne Glättung bestimmt; andere Ecken-Kriterien können anders ausfallen. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Kurven.** Mit $A = U\,\mathrm{diag}(\sigma_i)\,V^\top$, $\beta = U^\top b$ und den Filterfaktoren $f_i(\lambda) = \sigma_i^2/(\sigma_i^2 + \lambda^2)$ ist die Tikhonov-Lösung
$x_\lambda = \sum_i f_i\,(\beta_i/\sigma_i)\,v_i$. Das Residuum ist $\|A x_\lambda - b\|^2 = \sum_i (1 - f_i)^2\beta_i^2$, die Lösungsnorm $\|x_\lambda\|^2 = \sum_i (f_i\beta_i/\sigma_i)^2$ und die
Freiheitsgrade $\mathrm{df}(\lambda) = \sum_i f_i$.

**Diskrepanzprinzip.** $\lambda_{\mathrm{DP}} = \max\{\lambda : \|A x_\lambda - b\|^2 \le \tau^2\sigma^2 n\}$; das Residuum wächst mit $\lambda$, die Lösung ist also eindeutig.

**GCV.** $\lambda_{\mathrm{GCV}} = \arg\min_\lambda \|A x_\lambda - b\|^2/(n - \mathrm{df}(\lambda))^2$, äquivalent zur Kreuzvalidierung mit Weglassen eines Messwerts, gewichtet mit der Hat-Matrix
$H = A (A^\top A + \lambda^2 I)^{-1} A^\top$ und $\mathrm{tr}\,H = \mathrm{df}$.

**L-Kurve.** Mit $\rho = \log\|A x_\lambda - b\|$ und $\eta = \log\|x_\lambda\|$ als Funktionen von $t = \log\lambda$ ist die Krümmung
$\kappa = (\rho'\eta'' - \rho''\eta')/(\rho'^2 + \eta'^2)^{3/2}$; $\lambda_{\mathrm{L}}$ ist ihr Maximum.

**Maßstab.** Erwarteter Fehler $\sqrt{\sum_i [f_i(c_i + \rho_i) - c_i]^2 + f_i^2\sigma^2/\sigma_i^2}\,/\,\|x\|$ mit $c = V^\top x$ und dem Modellfehler $\rho_i = (U^\top r)_i/\sigma_i$ der ehrlichen
Daten; $\lambda^*$ ist sein Minimum über das Gitter. Das *Realisierungsorakel* minimiert den tatsächlichen Fehler $\|x_\lambda - x\|/\|x\|$ der einzelnen Stichprobe.

**Rauschschätzung.** $\hat\sigma = \sqrt{\mathrm{mean}_{i > n - m}\,\beta_i^2}$ mit $m = 8$; sie ist erwartungstreu, solange $\sigma_i c_i$ und der Modellfehler dort klein gegen $\sigma$ sind.

**Literatur.** Das Beispiel ist erfunden; das Diskrepanzprinzip geht auf V. A. Morozov (1966) zurück, GCV auf G. H. Golub, M. Heath und G. Wahba (1979), die L-Kurve auf
P. C. Hansen und D. P. O'Leary (1993); die beschriebenen Fehlmodi stehen bei P. C. Hansen, *Discrete Inverse Problems: Insight and Algorithms*, SIAM 2010
(alles aus dem Gedächtnis, nicht geprüft; die Fehlraten hier sind gemessen, nicht der Literatur entnommen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Inverse Probleme: vom Rauschen zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-inverse-probleme.html)."
)
