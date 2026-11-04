"""Carga, validación y transformación de una instantánea pública de PokéAPI."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

STATS = ["hp", "attack", "defense", "special_attack", "special_defense", "speed"]
FEATURES = STATS + ["log_height", "log_weight"]
LABELS = [
    "HP",
    "Ataque",
    "Defensa",
    "Ataque especial",
    "Defensa especial",
    "Velocidad",
    "log(1 + altura en m)",
    "log(1 + peso en kg)",
]


def load_data(root):
    """Especies predeterminadas, matrices de entrada y resumen del preprocesamiento.

    Los tipos se mantienen fuera de X para poder interpretar el mapa principal
    sin incluirlos en su distancia. X_types añade ese bloque en el experimento
    complementario, con una ponderación común para los 18 indicadores.
    """
    root = Path(root)
    meta = json.loads((root / "procedencia.json").read_text())
    for name, info in meta["files"].items():
        if hashlib.sha256((root / "raw" / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Integridad incorrecta: {name}")

    def read(name):
        """Lee una tabla de la copia local de PokéAPI."""
        return pd.read_csv(root / "raw" / name)

    raw = read("pokemon.csv")
    frame = raw.loc[raw.is_default.eq(1)].copy()
    if frame.id.duplicated().any() or frame.species_id.duplicated().any():
        raise ValueError("Hay duplicados de identificador o de especie")
    stats = read("pokemon_stats.csv")
    stats = stats[stats.stat_id.between(1, 6)].pivot(
        index="pokemon_id", columns="stat_id", values="base_stat"
    )
    stats.columns = STATS
    frame = frame.merge(stats, left_on="id", right_index=True, validate="one_to_one")
    species = read("pokemon_species.csv")[
        ["id", "generation_id", "is_legendary", "is_mythical"]
    ]
    frame = frame.merge(
        species,
        left_on="species_id",
        right_on="id",
        suffixes=("", "_species"),
        validate="one_to_one",
    )
    names = read("pokemon_species_names.csv")
    names = names[names.local_language_id.eq(9)][["pokemon_species_id", "name"]]
    frame = frame.merge(
        names, left_on="species_id", right_on="pokemon_species_id", validate="one_to_one"
    )
    types = read("pokemon_types.csv").merge(
        read("types.csv")[["id", "identifier"]], left_on="type_id", right_on="id"
    )
    types = types.pivot(index="pokemon_id", columns="slot", values="identifier").rename(
        columns={1: "type1", 2: "type2"}
    )
    frame = frame.merge(types, left_on="id", right_index=True, validate="one_to_one")
    if len(frame) != int(raw.is_default.sum()):
        raise ValueError("Se perdieron filas durante las uniones")
    if frame.type1.isna().any():
        raise ValueError("Falta el tipo primario de una especie")
    # La falta del segundo tipo expresa monotipo; no es un valor a imputar.
    frame["type2"] = frame.type2.fillna("")
    frame["height_m"] = frame.height / 10
    frame["weight_kg"] = frame.weight / 10
    numeric = STATS + ["height_m", "weight_kg"]
    missing = frame[numeric].isna().sum().to_dict()
    medians = frame[numeric].median()
    frame[numeric] = frame[numeric].fillna(medians)
    if (frame[numeric] <= 0).any().any():
        raise ValueError(
            "Se encontraron valores no positivos en atributos físicos/estadísticos"
        )
    frame["log_height"] = np.log1p(frame.height_m)
    frame["log_weight"] = np.log1p(frame.weight_kg)
    frame = frame.sort_values("species_id").reset_index(drop=True)
    # El mismo orden de especies se conserva en todas las matrices del análisis.
    mean, scale = frame[FEATURES].mean(), frame[FEATURES].std(ddof=0)
    if (scale == 0).any():
        raise ValueError("Hay una variable constante")
    X = ((frame[FEATURES] - mean) / scale).to_numpy()
    type_names = sorted(set(frame.type1) | (set(frame.type2) - {""}))
    T = np.array(
        [
            [int(t == a or t == b) for t in type_names]
            for a, b in zip(frame.type1, frame.type2)
        ],
        dtype=float,
    )
    p = T.mean(0)
    type_scale = float(np.sqrt(X.shape[1] / np.sum(p * (1 - p))))
    # Un único factor de bloque evita amplificar individualmente los tipos raros.
    X_types = np.column_stack([X, (T - p) * type_scale])
    audit = {
        "raw_rows": len(raw),
        "rows": len(frame),
        "excluded_forms": len(raw) - len(frame),
        "duplicate_ids": int(frame.id.duplicated().sum()),
        "missing_numeric": missing,
        "monotype": int(frame.type2.eq("").sum()),
        "dualtype": int(frame.type2.ne("").sum()),
        "missing_base_experience": int(frame.base_experience.isna().sum()),
        "features": FEATURES,
        "mean": mean.to_dict(),
        "scale": scale.to_dict(),
        "median": medians.to_dict(),
        "type_names": type_names,
        "type_mean": p.tolist(),
        "type_scale": type_scale,
        "source_commit": meta["commit"],
    }
    return frame, X, T, X_types, audit
