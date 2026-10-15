"""Konstanten: Regler-Grenzen, feste Annahmen, Experimente, Farben und Presets."""

# Regler. Rechenaufwand am Maximum: SVD einer 96 x 96-Matrix (O(n^3), unter 20 ms); Kurven über 361 Parameter, die Experimente nutzen feste Ströme
# mit 400 Wiederholungen (400 x 361 x 96 Koeffizienten, unter 1 s).
N_MIN, N_MAX, N_STEP, DEFAULT_N = 12, 96, 4, 48
W_MIN, W_MAX, W_STEP, DEFAULT_W = 0.01, 0.05, 0.005, 0.03        # Breite der Unschärfe (Anteil der Länge 1)
NOISE_CHOICES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)        # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-3
DATA_CHOICES = ("ehrlich", "verbrechen")
DATA_LABELS = {"ehrlich": "Ehrliche Daten (analytisch, vom Gitter unabhängig)", "verbrechen": "Daten aus demselben Gitter (inverse crime)"}
RULES = ("dp", "gcv", "lcurve")
RULE_LABELS = {"dp": "Diskrepanzprinzip (braucht σ)", "gcv": "Generalisierte Kreuzvalidierung (GCV)", "lcurve": "L-Kurve (Ecke der Krümmung)"}
RULE_SHORT = {"dp": "Diskrepanzprinzip", "gcv": "GCV", "lcurve": "L-Kurve", "oracle": "Orakel (Realisierung)"}
SIGMA_FACTORS = (0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 4.0)             # angenommenes σ als Vielfaches des wahren (nur Diskrepanzprinzip)
DEFAULT_SFACTOR = 1.0
TAU = 1.0                                                          # Sicherheitsfaktor des Diskrepanzprinzips
SEED_MAX, DEFAULT_SEED = 999999, 11

LAM_GRID = (1e-8, 10.0, 361)                                      # untere Grenze, obere Grenze, Zahl der Punkte (logarithmisch)
SINGULAR_COND = 1e15                                              # ab hier ist die Matrix in doppelter Genauigkeit numerisch singulär
SHOW_RANGE = (-1.5, 2.0)                                          # Wertebereich der Rekonstruktion im Diagramm; größere Werte werden abgeschnitten
FAIL_FACTOR = 2.0                                                 # „gescheitert“: Fehler über dem Doppelten des erwarteten Optimums
TAIL_M = 8                                                        # Zahl der hinteren Komponenten für die Rauschschätzung

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 970000
EXP_REPS = 400
EXP_NOISE_LEVELS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)                 # Experiment 2 und 3
EXP_N_SWEEP = (24, 32, 48, 64, 96)                                # Experiment 2
EXP_OPERATORS = ("integral", "integral2")                         # Experiment 4: andere Matrizen mit algebraisch fallenden Singulärwerten
OPERATOR_LABELS = {"integral": "Integration (σᵢ ~ 1/i)", "integral2": "doppelte Integration (σᵢ ~ 1/i²)"}
EXP_OP_NOISE = (1e-2, 1e-3)
GCV_BOUND_FACTORS = (1.0, 10.0)                                   # Experiment 4: GCV nur für λ ≥ Faktor · σ_n

COLOR_TRUTH = "#2F6B65"
COLOR_NAIVE = "#c0392b"
COLOR_DP = "#4c78a8"
COLOR_GCV = "#D68A2E"
COLOR_LC = "#8e44ad"
COLOR_ORACLE = "#2F6B65"
COLOR_GREY = "#8a8a8a"
RULE_COLORS = {"dp": COLOR_DP, "gcv": COLOR_GCV, "lcurve": COLOR_LC, "oracle": COLOR_ORACLE}

PRESETS = {
    "Standard": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "rule": "dp", "sfactor": 1.0, "seed": DEFAULT_SEED},
    "GCV scheitert": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "rule": "gcv", "sfactor": 1.0, "seed": 14},
    "L-Kurve zu klein": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "rule": "lcurve", "sfactor": 1.0, "seed": 4},
    "σ unterschätzt": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "rule": "dp", "sfactor": 0.25, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard": "Standardinstanz aus Stück 1 bis 3, Diskrepanzprinzip mit dem wahren σ: der Fehler liegt bei rund 12 %, nahe dem Optimum.",
    "GCV scheitert": "Eine Stichprobe (Seed 14), bei der GCV den Parameter am unteren Rand des Gitters wählt: das Rauschen kommt kaum gedämpft durch, der Fehler liegt bei rund 86 % statt 12 %.",
    "L-Kurve zu klein": "Eine Stichprobe (Seed 4): die Ecke der L-Kurve liegt beim 0,06-fachen des besten Parameters, die Lösung ist zu rau, der Fehler liegt bei rund 32 % statt 12 %.",
    "σ unterschätzt": "Das Diskrepanzprinzip mit einem um den Faktor 4 zu kleinen σ: es verlangt mehr Anpassung, als die Daten hergeben, und wählt λ zu klein.",
}
