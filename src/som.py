"""SOM online de Kohonen, implementado con NumPy (sin bibliotecas SOM).

Autores del proyecto: Sofía Muñiz Garzón y Alberto Caballero Lorente.
Referencia: kohonen01.pdf, diapositivas 7-13. Coordenadas (fila, columna),
índices desde cero y rejilla rectangular abierta, sin conexión entre bordes.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SOM:
    """Rejilla de prototipos con aprendizaje online y vecindad gaussiana."""

    rows: int = 12
    cols: int = 12
    epochs: int = 100
    alpha_start: float = 0.5
    alpha_end: float = 0.03
    sigma_start: float = 6.0
    sigma_end: float = 0.7
    seed: int = 42

    def __post_init__(self):
        """Parámetros válidos y distancias entre las posiciones de la rejilla."""
        if min(self.rows, self.cols, self.epochs) < 1:
            raise ValueError("Dimensiones y épocas deben ser positivas")
        if not 0 < self.alpha_end <= self.alpha_start <= 1:
            raise ValueError("Se requiere 0 < alpha_end <= alpha_start <= 1")
        if not 0 < self.sigma_end <= self.sigma_start:
            raise ValueError("Se requiere 0 < sigma_end <= sigma_start")
        self.coords = np.indices((self.rows, self.cols)).reshape(2, -1).T
        self.grid_d2 = ((self.coords[:, None] - self.coords[None, :]) ** 2).sum(2)

    @staticmethod
    def decay(start, end, fraction):
        """Decaimiento exponencial con extremos especificados exactamente."""
        return start * (end / start) ** fraction

    def initialize(self, X, rng):
        """Inicialización PCA; ruido pequeño evita empates exactos entre pesos."""
        _, s, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        # El signo queda fijado por la componente de mayor valor absoluto.
        for v in vt[:2]:
            if v[np.argmax(np.abs(v))] < 0:
                v *= -1
        axes = np.array(
            np.meshgrid(
                np.linspace(-1, 1, self.rows), np.linspace(-1, 1, self.cols), indexing="ij"
            )
        )
        axes = axes.reshape(2, -1).T
        self.weights = X.mean(0) + axes @ (vt[:2] * (s[:2] / np.sqrt(len(X)))[:, None])
        self.weights += rng.normal(0, 0.01, self.weights.shape)

    def winners(self, X, two=False):
        """BMU y opcionalmente segunda BMU; cálculo por lotes para limitar memoria."""
        answers = []
        for batch in np.array_split(np.asarray(X), max(1, int(np.ceil(len(X) / 512)))):
            d2 = ((batch[:, None, :] - self.weights[None, :, :]) ** 2).sum(2)
            answers.append(
                np.argsort(d2, axis=1, kind="stable")[:, :2]
                if two
                else np.argmin(d2, axis=1)
            )
        return np.concatenate(answers)

    def quality(self, X):
        """Cuantización, errores topográficos y ocupación de la rejilla."""
        first, second = self.winners(X, two=True).T
        delta = np.abs(self.coords[first] - self.coords[second])
        hits = np.bincount(first, minlength=len(self.weights))
        return {
            "qe": float(np.linalg.norm(X - self.weights[first], axis=1).mean()),
            "te8": float((delta.max(1) > 1).mean()),
            "te4": float((delta.sum(1) > 1).mean()),
            "occupied": int((hits > 0).sum()),
            "empty": int((hits == 0).sum()),
        }

    def fit(self, X):
        """Ajusta los pesos presentando todos los datos una vez por época.

        Cada fila de X es un Pokémon y cada columna, una variable preparada.
        La semilla fija tanto el ruido inicial como el orden de presentación.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] < 2 or len(X) < 2 or not np.isfinite(X).all():
            raise ValueError(
                "X debe ser una matriz finita con al menos dos filas y columnas"
            )
        rng = np.random.default_rng(self.seed)
        self.initialize(X, rng)
        self.history = [{"epoch": 0, **self.quality(X)}]
        total_steps = self.epochs * len(X)
        t = 0
        for epoch in range(1, self.epochs + 1):
            # Cada época presenta todas las observaciones, sin reemplazo.
            for i in rng.permutation(len(X)):
                x = X[i]
                c = np.argmin(((self.weights - x) ** 2).sum(1))
                f = t / max(1, total_steps - 1)
                alpha = self.decay(self.alpha_start, self.alpha_end, f)
                sigma = self.decay(self.sigma_start, self.sigma_end, f)
                h = np.exp(-self.grid_d2[c] / (2 * sigma * sigma))
                self.weights += alpha * h[:, None] * (x - self.weights)
                t += 1
            if epoch % 5 == 0 or epoch == self.epochs:
                self.history.append({"epoch": epoch, **self.quality(X)})
        return self

    def u_matrix(self):
        """Media de distancias euclidianas a vecinos ortogonales existentes.

        Los bordes usan 2 o 3 vecinos; no se introduce relleno con ceros.
        Se devuelve una U-matrix resumida por neurona (rows x cols).
        """
        result = np.zeros(len(self.weights))
        for i in range(len(self.weights)):
            adjacent = self.grid_d2[i] == 1
            result[i] = np.linalg.norm(
                self.weights[adjacent] - self.weights[i], axis=1
            ).mean()
        return result.reshape(self.rows, self.cols)
