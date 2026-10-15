# Parameterwahl ohne Wahrheit: GCV, Diskrepanzprinzip, L-Kurve – Streamlit-Demo

Viertes Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Es baut auf [schlecht-gestellt-demo](https://github.com/sebastian-hanisch/schlecht-gestellt-demo), [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo) und [tsvd-demo](https://github.com/sebastian-hanisch/tsvd-demo) auf und nutzt dieselbe Welt: ein Phantom aus zwei Hügeln und einem Kasten, gemessen mit einer Gauß-Unschärfe und weißem Rauschen. Die Wahrheit ist bekannt, weil das Phantom selbst die Wahrheit ist; sie dient hier nur als Maßstab, nie als Eingabe einer Regel.

## Kernfrage

In Stück 2 und 3 war das beste λ ein Orakel. Wie wählt man es aus den Daten allein? Drei Regeln stehen zur Wahl: das Diskrepanzprinzip (braucht σ), die generalisierte Kreuzvalidierung und die L-Kurve. Wie oft treffen sie, wie oft scheitern sie, und wovon hängt es ab?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie (Phantom, analytische Unschärfe, Gitter, Rauschen, zwei Datenarten):

- **Wahrheit** x(s) auf [0, 1]: zwei Gauß-Hügel und ein Kasten; **Messung** mit dem Gauß-Kern der Breite w, analytisch bekannt.
- **Gitter** mit n Punkten. Standardinstanz n = 48, w = 0,03, Rauschen 0,1 % des Effektivwerts (absolut 3,4·10⁻⁴), Kondition 1,26·10⁴.
- **Daten:** *ehrlich* (analytisch, vom Gitter unabhängig) oder *aus demselben Gitter* (b = A x + e, „inverse crime“); das Rauschen ist in beiden gleich.
- **Tikhonov** x_λ = V diag(σᵢ/(σᵢ² + λ²)) Uᵀ b über ein logarithmisches Gitter von 10⁻⁸ bis 10 (361 Punkte).

Alles ist erfunden; die Wahrheit ist das simulierte Phantom.

## Methodik

- **Diskrepanzprinzip:** größtes λ mit ‖A x_λ − b‖² ≤ τ²σ²n (τ = 1). Braucht σ.
- **GCV:** Minimum von ‖A x_λ − b‖²/(n − df)². Braucht kein σ.
- **L-Kurve:** Ecke der größten Krümmung der Kurve (log ‖A x_λ − b‖, log ‖x_λ‖) über log λ, per endlicher Differenzen auf dem Gitter. Braucht kein σ.
- **Maßstab:** das *beste erwartete λ\** (Minimum des geschlossen gerechneten erwarteten Fehlers) und das *Realisierungsorakel* (das λ mit dem kleinsten tatsächlichen Fehler der einzelnen Stichprobe). „Gescheitert“ heißt: Fehler über dem Doppelten des erwarteten Optimums.
- **Experimente** laufen über feste Zufallsströme (SplitMix64): 400 Wiederholungen je Zelle, nie ein einzelner Seed. Die Primäransicht ist eine einzelne Stichprobe und sagt nichts über die Häufigkeit des Scheiterns.

## Befunde (gemessen, keine Behauptungen)

Standardinstanz (n = 48, w = 0,03, Rauschen 0,1 %, ehrliche Daten), wenn nicht anders genannt. Jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund | Test |
|---|---|---|
| Wer trifft (Experiment 1, ehrliche Daten)? | λ\* = 0,0094, erwarteter Fehler 11,7 %. **Diskrepanzprinzip:** Median 11,9 %, 90. Perzentil 12,1 %, **kein** Scheitern; wählt λ im Median doppelt so groß wie λ\*. **GCV:** Median 11,8 %, aber **15 % gescheitert**, 90. Perzentil 63 %, RMS 42 %; das 5. Perzentil von λ/λ\* liegt bei 0 (Gitterrand). **L-Kurve:** Median 27 %, **68 %** gescheitert, λ im Median beim 0,06-Fachen von λ\*. Realisierungsorakel: 11,7 %. | `test_experiment_1_honest`, `test_experiment_1_quantiles_of_the_chosen_parameter` |
| Warum scheitert GCV? | In den gescheiterten Wiederholungen sind die hinteren Komponenten zufällig klein (Median **0,81 σ** gegen 1,00 σ sonst): die Daten sehen fast rauschfrei aus, und GCV wählt λ gegen null (56 % der Gescheiterten unter 1 % von λ\*). Der Fehler ist dann der der direkten Inversion dieser Stichprobe (Median 1,32 gegen 2,30 über alle Wiederholungen). | `test_gcv_failure_anatomy` |
| Und mit Daten aus demselben Gitter? | λ\* = 0,0071, Optimum 7,1 %. Diskrepanzprinzip 7,8 % (λ im Median das 2,4-Fache von λ\*), GCV Median 7,5 % mit **21 %** gescheitert (90. Perzentil 87 %), L-Kurve Median 24 % mit **90 %**. Orakel 7,0 %. | `test_experiment_1_crime` |
| Über Rauschstufen (Experiment 2)? | Anteil gescheitert für Rauschen 10 % / 1 % / 0,1 % / 0,01 % / 0,001 %, ehrliche Daten: **Diskrepanzprinzip 0 / 0 / 0 / 0 / 0 %, GCV 13 / 16 / 16 / 3 / 0 %, L-Kurve 25 / 71 / 60 / 0 / 0 %**. Daten aus demselben Gitter: Diskrepanzprinzip 0 / 0 / 0 / 0 / **99 %**, GCV 14 / 16 / 21 / 16 / 65 %, L-Kurve 28 / 85 / 92 / 0 / 90 %. | `test_experiment_2_failure_shares_by_noise` |
| Wo trägt das Diskrepanzprinzip nicht? | Bei 0,001 % Rauschen mit Daten aus demselben Gitter: Median 5,5 % statt 1,7 % für das Orakel; das Optimum (λ\* = 4,5·10⁻⁵) liegt dort unter der Wahl, die das Residuum erlaubt. | `test_experiment_2_median_errors_by_noise` |
| Hilft das Gitter? | n = 24 und 32: alle Regeln gleich dem Optimum (λ\* an der Untergrenze). n = 48: GCV 18 % gescheitert, L-Kurve 64 %; n = 64: GCV 14 %, **RMS das 383-Fache der Signalnorm** (einzelne Katastrophen); n = 96: GCV 3 %, L-Kurve 12 %. Diskrepanzprinzip: 9,6 % (n = 64) und 10,2 % (n = 96). | `test_experiment_2_grid` |
| Wie stark hängt das Diskrepanzprinzip an σ (Experiment 3)? | Angenommen / wahr = 0,25: RMS **82 %**, 80 % gescheitert; 0,5: 16,9 % (6 % gescheitert); 0,7: 11,8 %; 1: 11,9 %; 2: 12,2 %; 4: **12,6 %**. τ = 1 / 1,1 / 1,5 / 2: 11,9 / 11,9 / 12,0 / 12,2 %. Unterschätzen ist gefährlich, Überschätzen billig. | `test_experiment_3_assumed_sigma`, `test_experiment_3_tau` |
| Kann man σ schätzen? | σ̂ = Effektivwert der Beträge der letzten 8 Komponenten uᵢᵀb: Median σ̂/σ = 0,95 / 0,96 / 0,96 (Rauschen 10 % / 1 % / 0,1 %), 1,25 (0,01 %), **8,4** (0,001 %, ehrlich). Diskrepanzprinzip mit σ̂: 11,9 % bei 0,1 % (wie mit σ), 15,9 % statt 13,5 % bei 1 %, **50 % statt 19,5 %** bei 10 % (σ̂ streut bei nur acht Werten um rund ein Viertel). | `test_experiment_3_noise_estimate` |
| Andere Matrizen (Experiment 4)? | Integration (σᵢ ~ 1/i), 1 %: GCV RMS 27 % (3 % gescheitert), L-Kurve 54 % gescheitert; doppelte Integration, 1 %: GCV 13 % und 0,1 % GCV 17 % gescheitert. Das Diskrepanzprinzip liegt höchstens 17 % über dem Optimum (47 % statt 40 %). | `test_experiment_4_other_matrices` |
| Hilft eine Untergrenze für GCV? | Nein: Anteil gescheitert bei Rauschen 0,1 %, ehrliche Daten: ohne 14 %, λ ≥ σₙ 11 %, λ ≥ 10 σₙ 2,5 %; Daten aus demselben Gitter 23,5 % / 22 % / 12 %. Die Fehlminima liegen nicht nur am Rand. | `test_experiment_4_bounded_gcv` |
| Einzelstichproben (Presets) | Standard (Seed 11, Diskrepanzprinzip): 11,8 % beim 1,9-Fachen von λ\*, Orakel 11,8 %, direkte Inversion 237 %. GCV (Seed 14): **86 %**, λ praktisch null. L-Kurve (Seed 4): **32 %** beim 0,06-Fachen. σ auf ein Viertel (Seed 11): **83 %** beim 0,02-Fachen. | `test_standard_sample_and_presets` |
| Mini-Instanz von Hand | 3 × 3-Matrix (Singulärwerte 1 / 0,5 / 0,1), rauschfreie Daten: Diskrepanzprinzip mit σ = 0,01 gibt λ = 0,05 (Residuum² 2,998·10⁻⁴ gegen Ziel 3·10⁻⁴); GCV ist monoton steigend und wählt den kleinsten Wert; df(0,1) = 2,4516. | `test_mini_instance_by_hand_numbers`, `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **Zwei Orakel statt einem.** Der Plan nannte nur λ\*; die Messung trennt das erwartete Optimum vom Orakel der einzelnen Stichprobe (11,68 % gegen 11,65 %: fast gleich), das der faire Maßstab für eine Einzelstichprobe ist.
- **Der RMS-Fehler ist für GCV nicht aussagekräftig.** Einzelne Katastrophen (bei n = 64 das 383-Fache der Signalnorm) bestimmen ihn; die App zeigt Median, 90. Perzentil und den Anteil Gescheiterter.
- **Die Untergrenze repariert GCV nicht.** Die Erwartung „nur Gitterrand“ trägt nicht.
- **Rauschen schätzen löst nur halb.** Es trägt, solange die hinteren Komponenten reines Rauschen sind (0,1 % bis 1 %); bei sehr kleinem Rauschen ist es um das Achtfache zu hoch.
- **Gescheitert ist eine Konvention** (mehr als das Doppelte des erwarteten Optimums); die Anteile ändern sich mit ihr, die Rangfolge der Regeln nicht.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **σ ist bekannt (Diskrepanzprinzip)** | Ein um den Faktor 4 zu kleines σ liefert 82 % statt 12 %. | Bayes-Sicht |
| **Das Rauschen ist weiß und Gauß-verteilt** | Alle drei Regeln nehmen weißes Rauschen an. | – |
| **Ein Phantom, drei Matrizen, nur Tikhonov** | Die Rangwahl der abgeschnittenen SVD ist nicht gemessen. | – |
| **Das Orakel kennt die Wahrheit** | λ\* und das Realisierungsorakel sind in der Praxis nicht verfügbar. | – |
| **L-Kurve per endlicher Differenzen** | Die Krümmung ist ohne Glättung bestimmt; andere Ecken-Kriterien können anders ausfallen. | – |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo) und [tsvd-demo](https://github.com/sebastian-hanisch/tsvd-demo): Stück 2 und 3; dort sind λ\* und k\* Orakel, hier der Ersatz aus den Daten.
- [schlecht-gestellt-demo](https://github.com/sebastian-hanisch/schlecht-gestellt-demo): Stück 1; die Picard-Zahl dort ist ein verwandter, aber nicht gleicher Schätzer für den Rang.

## Tests

Mini-Instanz von Hand (3 × 3), Residuum, Lösungsnorm und df gegen `lstsq` und die explizite Hat-Matrix, GCV gegen ihre Definition, Diskrepanzprinzip gegen `scipy.optimize.brentq`, Krümmung gegen einen Kreis und eine weiche L-Form, Fehler gegen unabhängige Lösungen, Erwartung gegen Monte Carlo, bester Parameter gegen `minimize_scalar`, Generator (SplitMix64 gegen eine unabhängige Fassung), Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Slider-Grenzen, jede Rauschstufe, Regel und Datenart, σ-Faktoren, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Phantom, analytische Unschärfe, Operator, Daten) |
| `prw_model.py` | Kurven über das λ-Gitter, GCV, Diskrepanzprinzip, L-Kurve, Orakel, Rauschschätzung |
| `prw_evaluation.py` | Kennzahlen, Experimente, Verdikte |
| `prw_visualization.py`, `prw_pdf_export.py` | Figuren, PDF |
| `prw_constants.py`, `prw_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Iterative Verfahren** (Abbruch als Parameter): das Folgestück.
- **Bayes-Sicht** (σ und Prior als Teil des Modells): ein späteres Stück.
- **Weitere Regeln** (Quasi-Optimalität, unverzerrtes Risiko, Kreuzvalidierung): ändern die Aussage nicht.
- **Rangwahl der abgeschnittenen SVD mit denselben Regeln:** nicht gemessen.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
