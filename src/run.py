"""Experimento reproducible de Pokémon a partir de los CSV conservados localmente."""

import argparse
import json
import platform
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
from sklearn.manifold import trustworthiness
from data import load_data, FEATURES, STATS
from som import SOM
from plots import make_plots

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    """Registro legible de parámetros, métricas y datos auxiliares."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def macro_groups(model, X, k=6):
    """Resumen de los prototipos, ponderado por el número de especies que representan."""
    bmu = model.winners(X)
    hits = np.bincount(bmu, minlength=len(model.weights))
    km = KMeans(n_clusters=k, n_init=30, random_state=42).fit(
        model.weights, sample_weight=hits
    )
    # La numeración sigue la media de las seis estadísticas normalizadas.
    order = np.argsort(km.cluster_centers_[:, :6].mean(1))
    renumber = np.empty(k, dtype=int)
    renumber[order] = np.arange(1, k + 1)
    neurons = renumber[km.labels_]
    return neurons, neurons[bmu]


def type_summary(bmu, T, size):
    """Frecuencias de tipos por celda y cobertura del tipo más frecuente."""
    counts = np.zeros((size, T.shape[1]))
    np.add.at(counts, bmu, T)
    hits = np.bincount(bmu, minlength=size)
    # Un Pokémon de dos tipos contribuye a dos recuentos pero a un solo hit.
    dominant = counts.argmax(1)  # Empates: orden alfabético de tipos en inglés.
    purity = np.divide(counts.max(1), hits, out=np.zeros(size), where=hits > 0)
    score = float(counts.max(1).sum() / len(T))
    return counts, hits, dominant, purity, score


def main():
    """Ejecuta el análisis y conserva los resultados que se utilizan en el informe.

    X contiene las ocho variables numéricas, T los indicadores de tipo y XT
    la entrada del experimento mixto. Las mismas especies aparecen en las tres
    matrices y se mantienen en el mismo orden.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "resultados")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frame, X, T, XT, audit = load_data(ROOT / "data")
    write_json(out / "preprocesamiento.json", audit)
    frame.to_csv(out / "pokemon_limpio.csv", index=False)
    pd.DataFrame(X, columns=FEATURES).to_csv(
        out / "variables_normalizadas.csv", index=False
    )
    pd.DataFrame(T.astype(int), columns=audit["type_names"]).to_csv(
        out / "tipos_multihot.csv", index=False
    )
    frame[STATS + ["height_m", "weight_kg"]].describe().T.to_csv(out / "descriptivos.csv")
    experiments = []
    models = {}
    groups_by_seed = {}
    # Cada tamaño se prueba con las mismas tres semillas y cien épocas.
    for side in [8, 10, 12]:
        for seed in [7, 21, 42]:
            print(f"Entrenando SOM {side}x{side}, semilla {seed}...", flush=True)
            model = SOM(rows=side, cols=side, sigma_start=side / 2, seed=seed).fit(X)
            experiments.append(
                {"grid": f"{side}x{side}", "side": side, "seed": seed, **model.quality(X)}
            )
            models[(side, seed)] = model
            if side == 12:
                groups_by_seed[seed] = macro_groups(model, X)[1]
    exp = pd.DataFrame(experiments)
    exp.to_csv(out / "sensibilidad.csv", index=False)
    main_model = models[(12, 42)]
    bmu = main_model.winners(X)
    neurons, groups = macro_groups(main_model, X)
    counts, hits, dominant, purity, score = type_summary(bmu, T, 144)
    primary_counts = frame.type1.value_counts()
    baseline = float(primary_counts.max() / len(frame))
    rng = np.random.default_rng(2026)
    # La permutación mueve el par completo de tipos y deja las BMU fijas.
    null_scores = np.array(
        [type_summary(bmu, T[rng.permutation(len(T))], 144)[-1] for _ in range(200)]
    )
    null_p = float((1 + np.count_nonzero(null_scores >= score)) / (1 + len(null_scores)))
    kmeans = KMeans(n_clusters=6, n_init=30, random_state=42).fit(X)
    km_groups = kmeans.labels_
    print("Entrenando experimento con tipos (26 variables)...", flush=True)
    typed_model = SOM(seed=42).fit(XT)
    typed_bmu = typed_model.winners(XT)
    typed_neurons, typed_groups = macro_groups(typed_model, XT)
    typed_info = type_summary(typed_bmu, T, 144)
    coords = main_model.coords[bmu]
    quality = main_model.quality(X)
    metrics = {
        "n": len(frame),
        "dimensions": X.shape[1],
        "grid": [12, 12],
        "epochs": 100,
        "updates": 100 * len(frame),
        "seed": 42,
        "alpha": [0.5, 0.03],
        "sigma": [6.0, 0.7],
        **quality,
        "initial_qe": main_model.history[0]["qe"],
        "trustworthiness10": float(trustworthiness(X, coords, n_neighbors=10)),
        "dominant_type_coverage": score,
        "type_permutation_mean": float(null_scores.mean()),
        "type_permutation_std": float(null_scores.std(ddof=1)),
        "type_permutation_p": null_p,
        "global_primary_type_baseline": baseline,
        "silhouette_som6": float(silhouette_score(X, groups)),
        "silhouette_kmeans6": float(silhouette_score(X, km_groups)),
        "ari_som_kmeans": float(adjusted_rand_score(groups, km_groups)),
        "ami_primary_som": float(adjusted_mutual_info_score(frame.type1, groups)),
        "ami_primary_kmeans": float(adjusted_mutual_info_score(frame.type1, km_groups)),
        "stability_ari": {
            str(s): float(adjusted_rand_score(groups, groups_by_seed[s])) for s in [7, 21]
        },
        "typed": {
            "dimensions": XT.shape[1],
            **typed_model.quality(XT),
            "dominant_type_coverage": typed_info[-1],
            "ami_primary": float(adjusted_mutual_info_score(frame.type1, typed_groups)),
            "silhouette_numeric_space": float(silhouette_score(X, typed_groups)),
        },
        "versions": {
            x: importlib.metadata.version(x)
            for x in ["numpy", "pandas", "matplotlib", "scikit-learn", "scipy"]
        },
        "python": platform.python_version(),
    }
    write_json(out / "metricas.json", metrics)
    pd.DataFrame(main_model.history).to_csv(out / "entrenamiento.csv", index=False)
    assignments = frame.copy()
    assignments["bmu_fila"] = coords[:, 0]
    assignments["bmu_columna"] = coords[:, 1]
    assignments["grupo_som"] = groups
    assignments["error_cuantizacion"] = np.linalg.norm(X - main_model.weights[bmu], axis=1)
    assignments["tipo_dominante_neurona"] = np.array(audit["type_names"])[dominant[bmu]]
    assignments["proporcion_tipo_dominante"] = purity[bmu]
    assignments.to_csv(out / "asignaciones_pokemon.csv", index=False)
    profile = assignments.groupby("grupo_som")[STATS + ["height_m", "weight_kg"]].mean()
    profile.insert(0, "n", assignments.groupby("grupo_som").size())
    representatives = []
    for g in range(1, 7):
        ids = np.flatnonzero(groups == g)
        near = ids[np.argsort(np.linalg.norm(X[ids] - X[ids].mean(0), axis=1))[:3]]
        representatives.append(", ".join(frame.iloc[near].name))
    profile["representantes"] = representatives
    profile.to_csv(out / "perfiles_grupos.csv")
    neighbors = []
    # Los vecinos se buscan en ocho dimensiones, antes de consultar su celda.
    for name in ["Bulbasaur", "Charmander", "Pikachu", "Onix", "Snorlax", "Magikarp"]:
        i = int(frame.index[frame.name.eq(name)][0])
        distances = np.linalg.norm(X - X[i], axis=1)
        for j in np.argsort(distances, kind="stable")[1:4]:
            neighbors.append(
                {
                    "pokemon": name,
                    "vecino": frame.iloc[j]["name"],
                    "distancia_8D": float(distances[j]),
                    "distancia_rejilla": float(np.linalg.norm(coords[j] - coords[i])),
                    "bmu_origen": f"({coords[i,0]},{coords[i,1]})",
                    "bmu_vecino": f"({coords[j,0]},{coords[j,1]})",
                    "tipo_origen": "/".join(
                        t for t in [frame.iloc[i].type1, frame.iloc[i].type2] if t
                    ),
                    "tipo_vecino": "/".join(
                        t for t in [frame.iloc[j].type1, frame.iloc[j].type2] if t
                    ),
                }
            )
    pd.DataFrame(neighbors).to_csv(out / "vecinos.csv", index=False)
    neuron_table = pd.DataFrame(
        {
            "fila": main_model.coords[:, 0],
            "columna": main_model.coords[:, 1],
            "hits": hits,
            "u_matrix": main_model.u_matrix().ravel(),
            "tipo_dominante": [
                audit["type_names"][d] if h else "vacía" for d, h in zip(dominant, hits)
            ],
            "cobertura_tipo": purity,
            "grupo": neurons,
        }
    )
    neuron_table.to_csv(out / "neuronas.csv", index=False)
    np.savez_compressed(
        out / "modelo_som.npz",
        weights=main_model.weights,
        coords=main_model.coords,
        mean=np.array(list(audit["mean"].values())),
        scale=np.array(list(audit["scale"].values())),
        features=np.array(FEATURES),
        seed=42,
        rows=12,
        cols=12,
    )
    np.savez_compressed(
        out / "modelo_con_tipos.npz",
        weights=typed_model.weights,
        type_mean=audit["type_mean"],
        type_scale=audit["type_scale"],
    )
    write_json(
        out / "datos_graficos.json",
        {
            "u_matrix": main_model.u_matrix().tolist(),
            "hits": hits.reshape(12, 12).tolist(),
            "components": main_model.weights.reshape(12, 12, 8).tolist(),
            "groups": neurons.reshape(12, 12).tolist(),
            "dominant": dominant.reshape(12, 12).tolist(),
            "purity": purity.reshape(12, 12).tolist(),
            "types": audit["type_names"],
            "history": main_model.history,
        },
    )
    make_plots(
        out,
        frame,
        X,
        T,
        audit,
        main_model,
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
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
