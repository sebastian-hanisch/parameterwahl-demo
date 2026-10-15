"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 x 3-Matrix aus Stück 1 bis 3 (symmetrisch, Zeilensumme 1, Eigenwerte 1, 0,5, 0,1), Wahrheit x = (1, 2, 1), rauschfreie Daten b = A x = (1,3, 1,4, 1,3),
    Komponenten beta = U^T b = (-2,3094, 0, -0,08165). Handrechnung: Residuum^2(lam) = (lam^2/(1+lam^2))^2 * 5,3333 + (lam^2/(0,01+lam^2))^2 * 0,006667;
    bei lam = 0,05 ist es 2,998e-4, also ist lam = 0,05 die Lösung des Diskrepanzprinzips für sigma = 0,01 (Ziel sigma^2 n = 3e-4). GCV(lam) = Residuum^2 / (3 - df)^2
    ist für rauschfreie Daten monoton steigend (0,00653 bei 0,01; 0,00665 bei 0,05; 0,0073 bei 0,1); df(0,1) = 1/1,01 + 0,25/0,26 + 0,5 = 2,4516."""
    a = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
    x = np.array([1.0, 2.0, 1.0])
    b = np.array([1.3, 1.4, 1.3])
    return a, x, b
