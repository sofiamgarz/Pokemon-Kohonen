"""Pruebas contra cálculos independientes y propiedades del aprendizaje."""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from som import SOM
from data import load_data


class TestSOM(unittest.TestCase):
    """Comprobaciones pequeñas del cálculo y de la preparación de los datos."""

    def test_example_kohonen02(self):
        """El ejemplo manual conserva los pesos finales indicados en las diapositivas."""
        # Vecindad burbuja del ejemplo docente (no la gaussiana del proyecto).
        W = np.array([[1, 1.5], [2.5, 2], [4, 3.5], [5.5, 5.0]])
        patterns = np.array([[1, 1], [2, 2], [5, 4], [6, 5.0]])
        winners = []
        for x in patterns:
            c = np.argmin(np.linalg.norm(W - x, axis=1))
            winners.append(c)
            selected = np.abs(np.arange(4) - c) <= 1
            W[selected] += 0.5 * (x - W[selected])
        np.testing.assert_allclose(
            W, [[1.5, 1.625], [1.875, 1.75], [5, 4.1875], [5.625, 4.75]]
        )
        self.assertEqual(winners, [0, 1, 3, 3])
        # Distancias recalculadas: corrigen las aproximaciones de la diapositiva 12.
        distances = np.linalg.norm(patterns - W, axis=1)
        np.testing.assert_allclose(
            distances, [np.sqrt(0.640625), np.sqrt(0.078125), 0.1875, np.sqrt(0.203125)]
        )

    def test_gaussian_update(self):
        """La influencia disminuye con la distancia a la neurona ganadora."""
        s = SOM(2, 2, epochs=1)
        W = np.zeros((4, 2))
        x = np.array([2.0, 4.0])
        result = W + 0.5 * np.exp(-s.grid_d2[0] / 2)[:, None] * (x - W)
        np.testing.assert_allclose(result[0], [1, 2])
        np.testing.assert_allclose(result[1], np.array([1, 2]) * np.exp(-0.5))
        np.testing.assert_allclose(result[3], np.array([1, 2]) * np.exp(-1))

    def test_distances_and_topology(self):
        """Una rejilla de cuatro prototipos permite comprobar las distancias a mano."""
        s = SOM(2, 2)
        s.weights = np.array([[0, 0], [1, 0], [0, 1], [1, 1.0]])
        np.testing.assert_allclose(s.u_matrix(), 1)
        self.assertEqual(s.quality(s.weights)["qe"], 0)
        self.assertEqual(s.quality(s.weights)["occupied"], 4)

    def test_decay(self):
        """El decaimiento alcanza las tasas elegidas para el principio y el final."""
        self.assertAlmostEqual(SOM.decay(0.5, 0.03, 0), 0.5)
        self.assertAlmostEqual(SOM.decay(0.5, 0.03, 1), 0.03)

    def test_reproducibility_and_learning(self):
        """La misma semilla repite el ajuste de dos nubes de puntos sencillas."""
        rng = np.random.default_rng(4)
        X = np.vstack([rng.normal(-2, 0.2, (30, 2)), rng.normal(2, 0.2, (30, 2))])
        a = SOM(3, 3, epochs=30, sigma_start=1.5, sigma_end=0.4).fit(X)
        b = SOM(3, 3, epochs=30, sigma_start=1.5, sigma_end=0.4).fit(X)
        np.testing.assert_array_equal(a.weights, b.weights)
        self.assertLess(a.history[-1]["qe"], a.history[0]["qe"])

    def test_data_integrity_and_scaling(self):
        """Las especies, las escalas y el bloque de tipos mantienen lo previsto."""
        frame, X, T, XT, audit = load_data(Path(__file__).resolve().parents[1] / "data")
        self.assertEqual(len(frame), 1025)
        np.testing.assert_allclose(X.mean(0), 0, atol=1e-12)
        np.testing.assert_allclose(X.std(0), 1, atol=1e-12)
        self.assertTrue(np.isin(T.sum(1), [1, 2]).all())
        self.assertAlmostEqual(float(np.mean(np.sum(XT[:, 8:] ** 2, axis=1))), 8)


if __name__ == "__main__":
    unittest.main()
