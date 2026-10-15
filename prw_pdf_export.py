"""PDF-Export der aktuellen Instanz (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import prw_constants as C
from prw_evaluation import Analysis, de, err_pct, lam_text, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "σ": "sigma", "λ": "lambda", "τ": "tau", "−": "-", "€": "EUR", "≥": ">=", "·": "*", "⁻": "-", "⁰": "0", "¹": "1", "²": "2",
            "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
MAX_ROWS = 40


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Parameterwahl ohne Wahrheit"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    kind = "Daten aus demselben Gitter (inverse crime)" if s.data == "verbrechen" else "ehrliche Daten (analytisch)"
    pdf.multi_cell(0, 5, _s(f"Gitterpunkte n = {s.n}, Breite der Unschärfe w = {de(s.w, 3)}, Rauschen {de(100 * s.noise, 4)} % des Effektivwerts, {kind}, "
                            f"Regel: {C.RULE_SHORT[s.rule]}, angenommenes sigma = {de(s.sfactor, 2)} * wahres sigma, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Gewählt: lambda = {lam_text(a.lam_rule)} ({de(a.lam_rule / a.lam_star, 2)} * lambda*), Fehler dieser Stichprobe {err_pct(a.err_rule)}; "
        f"{'GESCHEITERT (mehr als das Doppelte des erwarteten Optimums)' if a.failed else 'nicht gescheitert'}.",
        f"Bester erwarteter Parameter (Wahrheit bekannt): lambda* = {lam_text(a.lam_star)} mit erwartetem Fehler {err_pct(a.err_star)}.",
        f"Orakel dieser Stichprobe: lambda = {lam_text(a.lam_oracle)}, Fehler {err_pct(a.err_oracle)}; direkte Inversion {err_pct(a.err_naive)}.",
        "Alle Regeln dieser Stichprobe: " + "; ".join(f"{C.RULE_SHORT[k]} {err_pct(a.pick_errors[k])} (lambda = {lam_text(a.lams[a.picks[k]])})" for k in ("dp", "gcv", "lcurve", "oracle")) + ".",
        f"Kondition der Matrix: {sci(a.cond)}; absolutes Rauschen sigma = {sci(a.sigma)}.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Punkt", "Ort s", "Wahrheit x", "Messwert b", "Gewählte Regel", "Orakel"], [16, 26, 30, 34, 38, 32]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(min(s.n, MAX_ROWS)):
        row = [f"{i + 1}", de(a.s[i], 3), de(a.x[i], 3), de(a.b[i], 4), de(a.x_rule[i], 3), de(a.x_oracle[i], 3)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    if s.n > MAX_ROWS:
        pdf.ln(1)
        pdf.cell(0, 5, _s(f"... {s.n - MAX_ROWS} weitere Zeilen nicht gezeigt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Tikhonov-Lösung mit Parameterwahl durch GCV, Diskrepanzprinzip oder L-Kurve. Quelle: parameterwahl-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
