"""Tablas del informe, dibujadas a partir de los resultados del experimento.

Las imágenes se guardan en resultados/tablas. Si el documento LaTeX está
disponible, también se actualizan sus referencias a las líneas del código.
"""

from pathlib import Path
import inspect
import json
import re
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from data import STATS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "resultados"
TABLES = RESULTS / "tablas"


def decimal(value, digits=3):
    """Coma decimal para las cifras de las tablas."""
    return f"{value:.{digits}f}".replace(".", ",")


def render_table(rows, widths, filename):
    """Tabla con reglas horizontales y espacio suficiente para cada celda.

    La tipografía serif y el fondo blanco siguen el formato del informe. La
    altura de cada fila depende del texto, para conservar las notas completas.
    """
    ratios = np.asarray(widths, dtype=float)
    ratios /= ratios.sum()
    wrapped, heights = [], []
    for index, row in enumerate(rows):
        font = FontProperties(
            family="DejaVu Serif", size=13, weight="bold" if index == 0 else "normal"
        )
        cells = []
        for value, ratio in zip(row, ratios):
            lines, current = [], ""
            for word in str(value).split():
                candidate = (current + " " + word).strip()
                width = TextPath((0, 0), candidate, prop=font).get_extents().width
                if current and width > ratio * 720 * 0.90:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            lines.append(current)
            cells.append("\n".join(lines))
        wrapped.append(cells)
        heights.append(0.235 * max(cell.count("\n") + 1 for cell in cells) + 0.16)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 13})
    total = sum(heights)
    fig, ax = plt.subplots(figsize=(10, total))
    ax.axis("off")
    table = ax.table(cellText=wrapped, colWidths=ratios, bbox=[0, 0, 1, 1], cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    for (row, col), cell in table.get_celld().items():
        cell.set_height(heights[row] / total)
        cell.set_facecolor("white")
        cell.visible_edges = ""
        cell.PAD = 0.035
        cell.get_text().set_color("black")
        if row == 0:
            cell.get_text().set_weight("bold")
    for y, thickness in [(1, 1.1), (1 - heights[0] / total, 0.7), (0, 1.1)]:
        ax.plot(
            [0, 1],
            [y, y],
            transform=ax.transAxes,
            color="black",
            linewidth=thickness,
            clip_on=False,
        )
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(TABLES / filename, dpi=260, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def descriptive_table():
    """Rangos y medianas de las variables antes de su transformación."""
    data = pd.read_csv(RESULTS / "descriptivos.csv", index_col=0)
    labels = [
        "HP",
        "Ataque",
        "Defensa",
        "Ataque especial",
        "Defensa especial",
        "Velocidad",
        "Altura (m)",
        "Peso (kg)",
    ]
    rows = [["Variable", "Mínimo", "Mediana", "Máximo"]]
    for column, label in zip(STATS + ["height_m", "weight_kg"], labels):
        rows.append(
            [label] + [decimal(data.loc[column, stat], 1) for stat in ["min", "50%", "max"]]
        )
    render_table(rows, [42, 19, 19, 20], "informe_descriptivos.png")


def configuration_table():
    """Decisiones de diseño de la ejecución principal y su justificación."""
    rows = [
        ["Parámetro", "Valor", "Justificación"],
        [
            "Rejilla",
            "12 × 12; rectangular abierta",
            "144 prototipos para representar gradientes; los bordes no se unen.",
        ],
        ["Distancia", "Euclidiana; 8 variables", "Atributos con escalas comparables."],
        [
            "Inicialización",
            "Dos componentes principales + ruido N(0; 0,01²)",
            "Aprovecha la variación principal y evita empates exactos.",
        ],
        [
            "Vecindad",
            "Gaussiana; σ de 6 a 0,7",
            "Organización amplia seguida de un ajuste local.",
        ],
        [
            "Aprendizaje",
            "α de 0,5 a 0,03",
            "Cambios más pequeños al final del entrenamiento.",
        ],
        [
            "Épocas",
            "100; 102.500 actualizaciones",
            "Presupuesto común para comparar rejillas.",
        ],
        ["Semilla", "42", "Fija el ruido inicial y el orden de presentación de los datos."],
    ]
    render_table(rows, [21, 31, 48], "informe_configuracion.png")


def sensitivity_table():
    """Media y dispersión muestral entre las tres semillas de cada tamaño."""
    runs = pd.read_csv(RESULTS / "sensibilidad.csv")
    rows = [["Rejilla", "QE", "TE, 8 vecinos (%)", "Ocupación"]]
    for side in [8, 10, 12]:
        selected = runs[runs.side.eq(side)]
        rows.append(
            [
                f"{side} × {side}",
                f"{decimal(selected.qe.mean())} ± {decimal(selected.qe.std())}",
                f"{decimal(100 * selected.te8.mean(), 2)} ± {decimal(100 * selected.te8.std(), 2)}",
                f"{selected.occupied.min()}–{selected.occupied.max()} / {side * side}",
            ]
        )
    render_table(rows, [16, 31, 31, 22], "informe_sensibilidad.png")


def profiles_table():
    """Tamaño y ejemplos de los seis perfiles calculados sobre el mapa."""
    profiles = pd.read_csv(RESULTS / "perfiles_grupos.csv")
    labels = [
        "Pequeños; estadísticas bajas",
        "Intermedios; ofensiva física moderada",
        "Rápidos; ofensiva especial",
        "Resistentes; velocidad baja",
        "Ofensiva física y velocidad",
        "Grandes, robustos y pesados",
    ]
    rows = [["Perfil", "n", "Características medias", "Ejemplos"]]
    for label, (_, row) in zip(labels, profiles.iterrows()):
        rows.append([f"G{int(row.grupo_som)}", int(row.n), label, row.representantes])
    render_table(rows, [9, 8, 43, 40], "informe_perfiles.png")


def neighbors_table():
    """Primer vecino de cada ejemplo en las ocho dimensiones de entrada."""
    neighbors = pd.read_csv(RESULTS / "vecinos.csv")
    rows = [["Pokémon", "Vecino más próximo", "Distancia", "BMU: origen / vecino"]]
    for _, row in neighbors.groupby("pokemon", sort=False).head(1).iterrows():
        rows.append(
            [
                row.pokemon,
                row.vecino,
                decimal(row.distancia_8D),
                row.bmu_origen + " / " + row.bmu_vecino,
            ]
        )
    render_table(rows, [21, 29, 15, 35], "informe_vecinos.png")


def manual_example_table():
    """Pesos y distancias del ejemplo docente, recalculados tras una pasada."""
    weights = np.array([[1, 1.5], [2.5, 2], [4, 3.5], [5.5, 5.0]])
    patterns = np.array([[1, 1], [2, 2], [5, 4], [6, 5.0]])
    for pattern in patterns:
        winner = np.argmin(np.linalg.norm(weights - pattern, axis=1))
        neighbors = np.abs(np.arange(4) - winner) <= 1
        weights[neighbors] += 0.5 * (pattern - weights[neighbors])
    rows = [["Patrón", "BMU final", "Peso de la BMU", "Distancia final"]]
    for pattern in patterns:
        distances = np.linalg.norm(weights - pattern, axis=1)
        winner = np.argmin(distances)
        vector = "(" + "; ".join(decimal(v, 4) for v in weights[winner]) + ")"
        rows.append(
            [
                f"({int(pattern[0])}; {int(pattern[1])})",
                f"N{winner + 1}",
                vector,
                decimal(distances[winner], 4),
            ]
        )
    render_table(rows, [18, 19, 39, 24], "informe_ejemplo_manual.png")


def comparison_table():
    """Silueta y relación con el tipo primario de las dos particiones numéricas."""
    metrics = json.loads((RESULTS / "metricas.json").read_text())
    rows = [
        ["Método, seis grupos", "Silueta en 8D", "AMI con tipo primario"],
        [
            "Perfiles del SOM",
            decimal(metrics["silhouette_som6"]),
            decimal(metrics["ami_primary_som"]),
        ],
        [
            "K-means sobre los Pokémon",
            decimal(metrics["silhouette_kmeans6"]),
            decimal(metrics["ami_primary_kmeans"]),
        ],
    ]
    render_table(rows, [45, 25, 30], "informe_comparacion.png")


def stability_table():
    """Variación del ajuste y de los seis perfiles entre semillas de la misma rejilla."""
    metrics = json.loads((RESULTS / "metricas.json").read_text())
    runs = pd.read_csv(RESULTS / "sensibilidad.csv")
    rows = [["Semilla", "QE", "TE, 8 vecinos (%)", "ARI respecto a 42"]]
    for seed in [7, 21, 42]:
        run = runs[runs.side.eq(12) & runs.seed.eq(seed)].iloc[0]
        agreement = metrics["stability_ari"].get(str(seed))
        rows.append(
            [
                seed,
                decimal(run.qe),
                decimal(100 * run.te8, 2),
                decimal(agreement) if agreement is not None else "Referencia",
            ]
        )
    render_table(rows, [16, 23, 29, 32], "informe_estabilidad.png")


def mixed_model_table():
    """Comparación del mapa numérico y el que añade los indicadores de tipo."""
    metrics = json.loads((RESULTS / "metricas.json").read_text())
    mixed = metrics["typed"]
    rows = [
        ["Medida", "SOM numérico", "SOM con tipos"],
        [
            "Cobertura del tipo dominante (%)",
            decimal(100 * metrics["dominant_type_coverage"], 2),
            decimal(100 * mixed["dominant_type_coverage"], 2),
        ],
        ["Neuronas ocupadas", f'{metrics["occupied"]}/144', f'{mixed["occupied"]}/144'],
        [
            "TE, 8 vecinos (%)",
            decimal(100 * metrics["te8"], 2),
            decimal(100 * mixed["te8"], 2),
        ],
        [
            "Silueta de seis perfiles en 8D",
            decimal(metrics["silhouette_som6"]),
            decimal(mixed["silhouette_numeric_space"]),
        ],
        [
            "AMI con el tipo primario",
            decimal(metrics["ami_primary_som"]),
            decimal(mixed["ami_primary"]),
        ],
    ]
    render_table(rows, [54, 23, 23], "informe_modelo_mixto.png")


def function_lines(function):
    """Intervalo real que permite localizar el cálculo de cada tabla."""
    lines, start = inspect.getsourcelines(function)
    return f"{start}--{start + len(lines) - 1}"


def figure_lines(name):
    """Bloque de Matplotlib que dibuja la figura identificada por su nombre."""
    lines = (ROOT / "src/plots.py").read_text().splitlines()
    end = next(
        i + 1 for i, line in enumerate(lines) if "save(fig," in line and f'"{name}"' in line
    )
    start = end
    while start > 1 and "plt.subplots(" not in lines[start - 1]:
        start -= 1
    return f"{start}--{end}"


def update_source_references():
    """Actualiza las líneas citadas cuando el documento LaTeX está disponible."""
    path = ROOT / "documentos/informe_pokemon.tex"
    if not path.is_file():
        return
    text = path.read_text()
    ranges = {
        "LineasDescriptivos": function_lines(descriptive_table),
        "LineasSensibilidad": function_lines(sensitivity_table),
        "LineasPerfiles": function_lines(profiles_table),
        "LineasVecinos": function_lines(neighbors_table),
        "LineasEjemplo": function_lines(manual_example_table),
        "LineasComparacion": function_lines(comparison_table),
        "LineasEstabilidad": function_lines(stability_table),
        "LineasMixto": function_lines(mixed_model_table),
        "LineasEntrenamiento": figure_lines("06_entrenamiento"),
        "LineasU": figure_lines("01_u_matrix"),
        "LineasHits": figure_lines("02_hit_map"),
        "LineasComponentes": figure_lines("03_componentes"),
        "LineasTipos": figure_lines("04_tipos_dominantes"),
        "LineasMapaPerfiles": figure_lines("08_perfiles"),
        "LineasCobertura": figure_lines("05_cobertura_tipos"),
        "LineasTiposMixto": figure_lines("10_componentes_tipos_mixto"),
        "LineasNumericosMixto": figure_lines("11_componentes_numericos_mixto"),
    }
    for name, interval in ranges.items():
        pattern = r"(\\newcommand\{\\" + name + r"\}\{)[^}]+(\})"
        text, count = re.subn(pattern, lambda m: m[1] + interval + m[2], text)
        if count != 1:
            raise ValueError(f"Referencia ausente o repetida: {name}")
    path.write_text(text)


def build():
    """Imágenes de las tablas y referencias sincronizadas con el código."""
    TABLES.mkdir(exist_ok=True)
    for table in [
        descriptive_table,
        configuration_table,
        sensitivity_table,
        profiles_table,
        neighbors_table,
        manual_example_table,
        comparison_table,
        stability_table,
        mixed_model_table,
    ]:
        table()
    update_source_references()
    print("Tablas guardadas en resultados/tablas.")


if __name__ == "__main__":
    build()
