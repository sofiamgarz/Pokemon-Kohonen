"""Visualizaciones científicas del SOM. Las coordenadas empiezan en cero."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from data import FEATURES, LABELS, STATS

TYPE_ES = {
    "normal": "Normal",
    "fighting": "Lucha",
    "flying": "Volador",
    "poison": "Veneno",
    "ground": "Tierra",
    "rock": "Roca",
    "bug": "Bicho",
    "ghost": "Fantasma",
    "steel": "Acero",
    "fire": "Fuego",
    "water": "Agua",
    "grass": "Planta",
    "electric": "Eléctrico",
    "psychic": "Psíquico",
    "ice": "Hielo",
    "dragon": "Dragón",
    "dark": "Siniestro",
    "fairy": "Hada",
}
TYPE_ABBR = {k: v[:3] for k, v in TYPE_ES.items()}
TYPE_COLORS = {
    "normal": "#9E9E76",
    "fighting": "#BA4438",
    "flying": "#A294D5",
    "poison": "#9542A3",
    "ground": "#CFAC63",
    "rock": "#A29242",
    "bug": "#98AB32",
    "ghost": "#655986",
    "steel": "#9C9CAF",
    "fire": "#EA7336",
    "water": "#477ED4",
    "grass": "#69A945",
    "electric": "#E8C135",
    "psychic": "#E75483",
    "ice": "#71BBBB",
    "dragon": "#7552CD",
    "dark": "#67574D",
    "fairy": "#D69AB3",
}


def make_plots(
    out,
    frame,
    X,
    T,
    audit,
    model,
    typed_model,
    bmu,
    groups,
    neurons,
    counts,
    hits,
    dominant,
    purity,
    exp,
    profile,
):
    """Mapas y resúmenes del experimento, con escalas comunes cuando son comparables."""
    figdir = Path(out) / "figuras"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )

    def save(fig, name):
        """La misma figura queda disponible como imagen y como PDF vectorial."""
        fig.savefig(figdir / (name + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(figdir / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    def grid(ax):
        """Ejes de la rejilla principal, con coordenadas de cero a once."""
        ax.set_xticks(range(12))
        ax.set_yticks(range(12))
        ax.set_xlabel("Columna")
        ax.set_ylabel("Fila")

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(model.u_matrix(), cmap="YlOrBr")
    grid(ax)
    ax.set_title("U-matrix: distancia media a los vecinos ortogonales", pad=14)
    fig.colorbar(im, ax=ax, label="Distancia euclidiana en el espacio estandarizado")
    save(fig, "01_u_matrix")
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    hm = hits.reshape(12, 12)
    im = ax.imshow(hm, cmap="Blues")
    for r, c in np.ndindex(hm.shape):
        ax.text(
            c,
            r,
            str(hm[r, c]),
            ha="center",
            va="center",
            fontsize=9,
            color="white" if hm[r, c] > hm.max() * 0.55 else "#152536",
        )
    grid(ax)
    ax.set_title("Hit map: Pokémon asignados a cada neurona", pad=14)
    fig.colorbar(im, ax=ax, label="Número de Pokémon")
    save(fig, "02_hit_map")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.7), layout="constrained")
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(
            model.weights[:, i].reshape(12, 12), cmap="RdBu_r", vmin=-2.5, vmax=2.5
        )
        ax.set_title(LABELS[i], fontsize=11)
        ax.set_xticks([0, 5, 11])
        ax.set_yticks([0, 5, 11])
    fig.colorbar(
        im,
        ax=axes,
        shrink=0.8,
        label="Peso del prototipo (desviaciones típicas; límites ±2,5)",
    )
    save(fig, "03_componentes")
    types = audit["type_names"]
    cmap = ListedColormap([TYPE_COLORS[t] for t in types])
    cmap.set_bad("#EEEEEE")
    data = np.ma.masked_where(hm == 0, dominant.reshape(12, 12))
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.imshow(data, cmap=cmap, vmin=-0.5, vmax=len(types) - 0.5)
    for k, (r, c) in enumerate(model.coords):
        text = TYPE_ABBR[types[dominant[k]]] if hits[k] else "-"
        ax.text(
            c,
            r,
            text,
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.68, "edgecolor": "none", "pad": 1.1},
        )
    grid(ax)
    ax.set_title("Tipo dominante por neurona (considerando ambos tipos)", pad=14)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=TYPE_COLORS[t], label=f"{TYPE_ABBR[t]}: {TYPE_ES[t]}")
            for t in types
        ],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=9,
    )
    save(fig, "04_tipos_dominantes")
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(
        np.ma.masked_where(hm == 0, purity.reshape(12, 12)), vmin=0, vmax=1, cmap="viridis"
    )
    grid(ax)
    ax.set_title("Proporción de Pokémon que poseen el tipo dominante", pad=14)
    fig.colorbar(im, ax=ax, label="Proporción dentro de la neurona")
    save(fig, "05_cobertura_tipos")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    hist = pd.DataFrame(model.history)
    axes[0].plot(hist.epoch, hist.qe, color="#B8423A", linewidth=2)
    axes[0].set(
        xlabel="Época", ylabel="Error de cuantización (QE)", title="Ajuste a los datos"
    )
    axes[1].plot(hist.epoch, hist.te8, color="#2D6F7D", linewidth=2, label="8 vecinos")
    axes[1].plot(hist.epoch, hist.te4, color="#999999", linestyle="--", label="4 vecinos")
    axes[1].set(
        xlabel="Época",
        ylabel="Error topográfico (TE)",
        title="Continuidad local de la rejilla",
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    save(fig, "06_entrenamiento")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, metric, title in [
        (axes[0], "qe", "Cuantización"),
        (axes[1], "te8", "Error topográfico (8 vecinos)"),
    ]:
        for seed, color in zip([7, 21, 42], ["#777777", "#2D6F7D", "#B8423A"]):
            sub = exp[exp.seed == seed]
            ax.plot(sub.side, sub[metric], "o-", label=f"Semilla {seed}", color=color)
        ax.set_xticks([8, 10, 12], ["8 × 8", "10 × 10", "12 × 12"])
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.2)
    save(fig, "07_sensibilidad")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
        layout="constrained",
        gridspec_kw={"width_ratios": [1, 1.3]},
    )
    cmap6 = ListedColormap(
        ["#9AB7A6", "#DFC16F", "#7396BB", "#BC849E", "#8E88AA", "#B85C50"]
    )
    axes[0].imshow(neurons.reshape(12, 12), vmin=0.5, vmax=6.5, cmap=cmap6)
    for k, (r, c) in enumerate(model.coords):
        axes[0].text(c, r, str(neurons[k]), ha="center", va="center", fontsize=9)
    axes[0].set(title="Seis perfiles de prototipos", xlabel="Columna", ylabel="Fila")
    z = pd.DataFrame(X, columns=FEATURES).assign(grupo=groups).groupby("grupo").mean()
    im = axes[1].imshow(z, vmin=-1.8, vmax=1.8, cmap="RdBu_r", aspect="auto")
    axes[1].set_xticks(
        range(8),
        ["HP", "ATQ", "DEF", "A.ESP", "D.ESP", "VEL", "ALT", "PESO"],
        rotation=45,
        ha="right",
    )
    axes[1].set_yticks(
        range(6), [f"G{g} (n={int(profile.loc[g,'n'])})" for g in range(1, 7)]
    )
    axes[1].set_title("Media de los Pokémon por perfil")
    fig.colorbar(im, ax=axes[1], label="Media estandarizada")
    save(fig, "08_perfiles")
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    freq = T.sum(0)
    order = np.argsort(freq)[::-1]
    ax.bar(
        [TYPE_ES[types[i]] for i in order],
        freq[order],
        color=[TYPE_COLORS[types[i]] for i in order],
    )
    ax.tick_params(axis="x", rotation=50)
    ax.set(
        ylabel="Pokémon que poseen el tipo",
        title="Distribución de los tipos (los Pokémon duales cuentan dos veces)",
    )
    save(fig, "09_distribucion_tipos")
    # El escalado se deshace para leer los componentes de tipo en su escala original.
    probs = typed_model.weights[:, 8:] / audit["type_scale"] + np.array(audit["type_mean"])
    fig, axes = plt.subplots(3, 6, figsize=(14, 7.2), layout="constrained")
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(probs[:, i].reshape(12, 12), cmap="Blues", vmin=0, vmax=1)
        ax.set_title(TYPE_ES[types[i]], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.65, label="Componente de tipo en escala original")
    save(fig, "10_componentes_tipos_mixto")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.7), layout="constrained")
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(
            typed_model.weights[:, i].reshape(12, 12), cmap="RdBu_r", vmin=-2.5, vmax=2.5
        )
        ax.set_title(LABELS[i], fontsize=11)
        ax.set_xticks([0, 5, 11])
        ax.set_yticks([0, 5, 11])
    fig.colorbar(
        im,
        ax=axes,
        shrink=0.8,
        label="Pesos numéricos del SOM mixto (desviaciones típicas)",
    )
    save(fig, "11_componentes_numericos_mixto")
