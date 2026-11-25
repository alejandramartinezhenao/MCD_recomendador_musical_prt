"""Pruebas rápidas de la Fase II (no requieren audio ni GPU)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fase2_audio import AST_SR, split_segments  # noqa: E402
from fase2_modelos import evaluate_recommender, quadrant_of  # noqa: E402


class CuadrantesTests(unittest.TestCase):
    def test_cuadrantes(self) -> None:
        got = quadrant_of([2, 7, 7, 2], [2, 7, 2, 7]).tolist()
        self.assertEqual(got, ["Negativa / baja", "Positiva / alta", "Positiva / baja", "Negativa / alta"])


class SegmentosTests(unittest.TestCase):
    def test_maximo_seis_fragmentos_de_10s(self) -> None:
        segs = split_segments(np.zeros(AST_SR * 200))
        self.assertEqual(len(segs), 6)
        self.assertTrue(all(len(s) == AST_SR * 10 for s in segs))

    def test_audio_corto_un_fragmento(self) -> None:
        self.assertEqual(len(split_segments(np.zeros(AST_SR * 5))), 1)


class RecomendadorTests(unittest.TestCase):
    def test_coordenadas_reales_superan_al_azar(self) -> None:
        rng = np.random.default_rng(0)
        coords = pd.DataFrame(rng.uniform(1, 9, (300, 2)), columns=["valencia", "activacion"])
        tabla = evaluate_recommender(coords, {"perfecto": coords}, k=5, n_random=20).set_index("sistema")
        self.assertAlmostEqual(tabla.loc["perfecto", "acierto_cuadrante@5"], 1.0)
        self.assertGreater(tabla.loc["perfecto", "acierto_cuadrante@5"], tabla.loc["Aleatorio", "acierto_cuadrante@5"])


if __name__ == "__main__":
    unittest.main()
