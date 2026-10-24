"""Fase I: perfiles de anotación, clasificador y prototipo de recomendador emocional.

Ejecución desde la terminal:

    python src/perfiles_recomendador.py
    python src/perfiles_recomendador.py --emocion triste
    python src/perfiles_recomendador.py --emocion alegre --grupo 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisis_deam import COLORS, compute_analysis, load_deam


RANDOM_STATE = 42
MIN_RATINGS_PER_RATER = 10
K_CANDIDATES = range(2, 7)

PROFILE_FEATURES = [
    "valencia_media",
    "activacion_media",
    "valencia_sd",
    "activacion_sd",
    "sesgo_valencia",
    "sesgo_activacion",
    "discrepancia",
]

FEATURE_LABELS = {
    "valencia_media": "Valencia media",
    "activacion_media": "Activación media",
    "valencia_sd": "Dispersión valencia",
    "activacion_sd": "Dispersión activación",
    "sesgo_valencia": "Sesgo valencia",
    "sesgo_activacion": "Sesgo activación",
    "discrepancia": "Discrepancia con consenso",
}

# Frase que describe un grupo cuando su centroide se aleja del promedio
# en una variable (z > 0 / z < 0).
FEATURE_PHRASES = {
    "valencia_media": ("valencias altas", "valencias bajas"),
    "activacion_media": ("activaciones altas", "activaciones bajas"),
    "valencia_sd": ("respuestas de valencia variadas", "respuestas de valencia concentradas"),
    "activacion_sd": ("respuestas de activación variadas", "respuestas de activación concentradas"),
    "sesgo_valencia": ("perciben más positividad", "perciben más negatividad"),
    "sesgo_activacion": ("perciben más energía", "perciben menos energía"),
    "discrepancia": ("se alejan del consenso", "cercanos al consenso"),
}

# Punto de referencia (valencia, activación) en la escala 1-9 para cada emoción.
# Ubicaciones orientativas según el modelo circumplejo de Russell (1980).
EMOTIONS = {
    "alegre": (7.0, 6.5),
    "energico": (6.5, 7.5),
    "relajado": (6.5, 3.5),
    "tranquilo": (5.5, 2.5),
    "triste": (3.0, 3.0),
    "melancolico": (4.0, 3.5),
    "tenso": (3.5, 7.0),
    "enojado": (2.5, 7.5),
}

GROUP_COLORS = [
    COLORS["blue"],
    COLORS["coral"],
    COLORS["teal"],
    COLORS["mustard"],
    COLORS["violet"],
    COLORS["navy"],
]


# ---------------------------------------------------------------------------
# 1. Perfiles de anotación
# ---------------------------------------------------------------------------


def build_rater_profiles(individual: pd.DataFrame) -> pd.DataFrame:
    """Resume a cada evaluador con variables de valencia y activación.

    El sesgo compara cada valoración con el consenso de los demás evaluadores
    de la misma canción (promedio sin incluir al propio evaluador).
    """
    ratings = individual.copy()
    for dim in ["valence", "arousal"]:
        grouped = ratings.groupby("song_id")[dim]
        song_sum = grouped.transform("sum")
        song_count = grouped.transform("count")
        ratings[f"{dim}_consensus"] = (song_sum - ratings[dim]) / (song_count - 1)
    ratings = ratings[ratings["valence_consensus"].notna()]
    ratings = ratings.replace([np.inf, -np.inf], np.nan).dropna()

    ratings["valence_diff"] = ratings["valence"] - ratings["valence_consensus"]
    ratings["arousal_diff"] = ratings["arousal"] - ratings["arousal_consensus"]
    ratings["abs_diff"] = (ratings["valence_diff"].abs() + ratings["arousal_diff"].abs()) / 2

    profiles = ratings.groupby("worker_id").agg(
        n_canciones=("song_id", "nunique"),
        valencia_media=("valence", "mean"),
        activacion_media=("arousal", "mean"),
        valencia_sd=("valence", "std"),
        activacion_sd=("arousal", "std"),
        sesgo_valencia=("valence_diff", "mean"),
        sesgo_activacion=("arousal_diff", "mean"),
        discrepancia=("abs_diff", "mean"),
    )
    profiles = profiles[profiles["n_canciones"] >= MIN_RATINGS_PER_RATER]
    return profiles.reset_index()


def describe_group(centroid_z: pd.Series) -> str:
    strongest = centroid_z.abs().sort_values(ascending=False)
    phrases = []
    for feature in strongest.index[:2]:
        if abs(centroid_z[feature]) < 0.5:
            continue
        positive, negative = FEATURE_PHRASES[feature]
        phrases.append(positive if centroid_z[feature] > 0 else negative)
    return ", ".join(phrases).capitalize() if phrases else "Perfil promedio"


def cluster_profiles(profiles: pd.DataFrame) -> dict[str, object]:
    scaler = StandardScaler()
    features_z = scaler.fit_transform(profiles[PROFILE_FEATURES])

    selection = []
    for k in K_CANDIDATES:
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(features_z)
        selection.append(
            {
                "k": k,
                "inercia": float(model.inertia_),
                "silhouette": float(silhouette_score(features_z, labels)),
            }
        )
    selection = pd.DataFrame(selection)
    best_k = int(selection.loc[selection["silhouette"].idxmax(), "k"])

    model = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
    raw_labels = model.fit_predict(features_z)

    # Numera los grupos del más grande al más pequeño para que la salida sea estable.
    order = pd.Series(raw_labels).value_counts().index.tolist()
    relabel = {old: new + 1 for new, old in enumerate(order)}
    profiles = profiles.copy()
    profiles["grupo"] = [relabel[label] for label in raw_labels]

    centroids_z = pd.DataFrame(
        model.cluster_centers_[order], columns=PROFILE_FEATURES, index=range(1, best_k + 1)
    )
    summary = profiles.groupby("grupo")[PROFILE_FEATURES].mean()
    summary.insert(0, "evaluadores", profiles["grupo"].value_counts().sort_index())
    summary.insert(
        0, "descripcion", [describe_group(centroids_z.loc[g]) for g in summary.index]
    )

    return {
        "profiles": profiles,
        "selection": selection,
        "best_k": best_k,
        "centroids_z": centroids_z,
        "summary": summary,
        "silhouette": float(selection.loc[selection["k"] == best_k, "silhouette"].iloc[0]),
    }


# ---------------------------------------------------------------------------
# 2. Clasificador de perfiles
# ---------------------------------------------------------------------------


def train_classifier(profiles: pd.DataFrame) -> dict[str, object]:
    x = profiles[PROFILE_FEATURES]
    y = profiles["grupo"]
    n_splits = int(min(5, y.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    predicted = cross_val_predict(tree, x, y, cv=cv)
    baseline = cross_val_predict(DummyClassifier(strategy="most_frequent"), x, y, cv=cv)

    tree.fit(x, y)
    importances = pd.Series(tree.feature_importances_, index=PROFILE_FEATURES).sort_values(
        ascending=False
    )
    labels = sorted(y.unique())

    return {
        "model": tree,
        "n_splits": n_splits,
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "baseline_accuracy": float(accuracy_score(y, baseline)),
        "confusion": pd.DataFrame(
            confusion_matrix(y, predicted, labels=labels), index=labels, columns=labels
        ),
        "importances": importances,
        "rules": export_text(tree, feature_names=PROFILE_FEATURES, decimals=2),
    }


# ---------------------------------------------------------------------------
# 3. Prototipo de recomendador emocional
# ---------------------------------------------------------------------------


def build_catalog(static: pd.DataFrame) -> pd.DataFrame:
    catalog = static[
        ["song_id", "valence_mean", "arousal_mean", "valence_std", "arousal_std", "quadrant"]
    ].copy()
    catalog["incertidumbre"] = (catalog["valence_std"] + catalog["arousal_std"]) / 2
    return catalog


def recommend(
    catalog: pd.DataFrame,
    valence: float,
    arousal: float,
    k: int = 5,
    bias: tuple[float, float] = (0.0, 0.0),
    uncertainty_weight: float = 0.3,
    exclude: set[int] | None = None,
) -> pd.DataFrame:
    """Devuelve las k canciones más cercanas a un punto del espacio emocional.

    - `bias` ajusta la emoción de cada canción a la percepción típica de un grupo
      de oyentes (sesgo medio de valencia y activación del grupo).
    - `uncertainty_weight` penaliza canciones con poco acuerdo entre evaluadores.
    """
    candidates = catalog if not exclude else catalog[~catalog["song_id"].isin(exclude)]
    candidates = candidates.copy()
    candidates["valencia_percibida"] = candidates["valence_mean"] + bias[0]
    candidates["activacion_percibida"] = candidates["arousal_mean"] + bias[1]
    candidates["distancia"] = np.hypot(
        candidates["valencia_percibida"] - valence,
        candidates["activacion_percibida"] - arousal,
    )
    candidates["puntaje"] = (
        candidates["distancia"] + uncertainty_weight * candidates["incertidumbre"]
    )
    columns = [
        "song_id",
        "valencia_percibida",
        "activacion_percibida",
        "quadrant",
        "distancia",
        "incertidumbre",
        "puntaje",
    ]
    return candidates.nsmallest(k, "puntaje")[columns].reset_index(drop=True)


def recommend_emotion(
    catalog: pd.DataFrame, emotion: str, k: int = 5, bias: tuple[float, float] = (0.0, 0.0)
) -> pd.DataFrame:
    valence, arousal = EMOTIONS[emotion]
    return recommend(catalog, valence, arousal, k=k, bias=bias)


def emotional_route(
    catalog: pd.DataFrame, start: str, end: str, steps: int = 6
) -> pd.DataFrame:
    """Lista de canciones que lleva gradualmente de una emoción a otra."""
    start_point = np.array(EMOTIONS[start])
    end_point = np.array(EMOTIONS[end])
    used: set[int] = set()
    route = []
    for step, t in enumerate(np.linspace(0, 1, steps), start=1):
        target = start_point + t * (end_point - start_point)
        song = recommend(catalog, target[0], target[1], k=1, exclude=used).iloc[0]
        used.add(int(song["song_id"]))
        route.append(
            {
                "paso": step,
                "valencia_objetivo": target[0],
                "activacion_objetivo": target[1],
                **song.to_dict(),
            }
        )
    route = pd.DataFrame(route)
    route["song_id"] = route["song_id"].astype(int)
    return route


def group_biases(cluster: dict[str, object]) -> dict[int, tuple[float, float]]:
    summary = cluster["summary"]
    return {
        int(group): (float(row["sesgo_valencia"]), float(row["sesgo_activacion"]))
        for group, row in summary.iterrows()
    }


# ---------------------------------------------------------------------------
# Ejecución, validación y salidas
# ---------------------------------------------------------------------------


def compute_phase1(analysis: dict[str, object]) -> dict[str, object]:
    profiles = build_rater_profiles(analysis["individual"])
    cluster = cluster_profiles(profiles)
    classifier = train_classifier(cluster["profiles"])
    catalog = build_catalog(analysis["static"])
    biases = group_biases(cluster)

    examples = []
    for emotion in EMOTIONS:
        recs = recommend_emotion(catalog, emotion)
        recs.insert(0, "emocion", emotion)
        examples.append(recs)
    examples = pd.concat(examples, ignore_index=True)

    route = emotional_route(catalog, "triste", "alegre")

    metrics = {
        "evaluadores_totales": int(analysis["individual"]["worker_id"].nunique()),
        "evaluadores_con_perfil": int(len(cluster["profiles"])),
        "minimo_canciones_por_evaluador": MIN_RATINGS_PER_RATER,
        "k_elegido": cluster["best_k"],
        "silhouette": cluster["silhouette"],
        "clasificador": "Árbol de decisión (profundidad máxima 3)",
        "validacion_cruzada_folds": classifier["n_splits"],
        "accuracy": classifier["accuracy"],
        "balanced_accuracy": classifier["balanced_accuracy"],
        "accuracy_linea_base": classifier["baseline_accuracy"],
        "sesgo_por_grupo": {
            str(g): {"valencia": b[0], "activacion": b[1]} for g, b in biases.items()
        },
    }

    return {
        "cluster": cluster,
        "classifier": classifier,
        "catalog": catalog,
        "biases": biases,
        "examples": examples,
        "route": route,
        "metrics": metrics,
    }


def validate_phase1(phase1: dict[str, object]) -> None:
    metrics = phase1["metrics"]
    assert metrics["evaluadores_con_perfil"] == 116
    assert metrics["k_elegido"] in K_CANDIDATES
    assert metrics["accuracy"] > metrics["accuracy_linea_base"]
    assert len(phase1["route"]) == phase1["route"]["song_id"].nunique()


def save_phase1_tables(phase1: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster = phase1["cluster"]
    cluster["profiles"].to_csv(output_dir / "perfiles_evaluadores.csv", index=False)
    cluster["summary"].reset_index().to_csv(output_dir / "resumen_grupos.csv", index=False)
    cluster["selection"].to_csv(output_dir / "seleccion_k.csv", index=False)
    phase1["examples"].to_csv(output_dir / "recomendaciones_por_emocion.csv", index=False)
    phase1["route"].to_csv(output_dir / "ruta_emocional_triste_alegre.csv", index=False)
    (output_dir / "reglas_clasificador.txt").write_text(
        phase1["classifier"]["rules"], encoding="utf-8"
    )
    (output_dir / "metricas_fase1.json").write_text(
        json.dumps(phase1["metrics"], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_phase1_plots(phase1: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster = phase1["cluster"]
    classifier = phase1["classifier"]
    catalog = phase1["catalog"]
    profiles = cluster["profiles"]
    selection = cluster["selection"]
    best_k = cluster["best_k"]

    # 05. Selección del número de grupos
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(selection["k"], selection["inercia"], marker="o", color=COLORS["navy"])
    axes[0].set_title("Método del codo")
    axes[0].set_xlabel("Número de grupos (k)")
    axes[0].set_ylabel("Inercia")
    axes[1].plot(selection["k"], selection["silhouette"], marker="o", color=COLORS["teal"])
    axes[1].axvline(best_k, color=COLORS["coral"], linestyle="--", linewidth=1)
    axes[1].set_title(f"Coeficiente silhouette (k elegido = {best_k})")
    axes[1].set_xlabel("Número de grupos (k)")
    axes[1].set_ylabel("Silhouette")
    for axis in axes:
        axis.set_xticks(list(K_CANDIDATES))
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "05_seleccion_numero_grupos.png", dpi=180)
    plt.close(fig)

    # 06. Perfiles de anotación
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    for group in sorted(profiles["grupo"].unique()):
        subset = profiles[profiles["grupo"] == group]
        axes[0].scatter(
            subset["sesgo_valencia"],
            subset["sesgo_activacion"],
            s=np.clip(subset["n_canciones"], 15, 200),
            alpha=0.7,
            color=GROUP_COLORS[group - 1],
            label=f"Grupo {group} (n={len(subset)})",
        )
    axes[0].axvline(0, color="#657485", linestyle="--", linewidth=1)
    axes[0].axhline(0, color="#657485", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Sesgo de valencia (vs. consenso)")
    axes[0].set_ylabel("Sesgo de activación (vs. consenso)")
    axes[0].set_title("Evaluadores según su sesgo de percepción")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    centroids = cluster["centroids_z"]
    limit = max(1.0, float(np.abs(centroids.values).max()))
    image = axes[1].imshow(centroids.values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axes[1].set_xticks(range(len(PROFILE_FEATURES)))
    axes[1].set_xticklabels(
        [FEATURE_LABELS[f] for f in PROFILE_FEATURES], rotation=35, ha="right"
    )
    axes[1].set_yticks(range(len(centroids)))
    axes[1].set_yticklabels([f"Grupo {g}" for g in centroids.index])
    for i in range(centroids.shape[0]):
        for j in range(centroids.shape[1]):
            value = centroids.iloc[i, j]
            axes[1].text(
                j, i, f"{value:.1f}", ha="center", va="center", fontsize=9,
                color="white" if abs(value) > 0.6 * limit else "black",
            )
    axes[1].set_title("Centroides estandarizados (z)")
    fig.colorbar(image, ax=axes[1], fraction=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "06_perfiles_anotacion.png", dpi=180)
    plt.close(fig)

    # 07. Clasificador
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    confusion = classifier["confusion"]
    axes[0].imshow(confusion.values, cmap="Blues")
    axes[0].set_xticks(range(len(confusion)))
    axes[0].set_xticklabels([f"G{g}" for g in confusion.columns])
    axes[0].set_yticks(range(len(confusion)))
    axes[0].set_yticklabels([f"G{g}" for g in confusion.index])
    threshold = confusion.values.max() / 2
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = confusion.iloc[i, j]
            axes[0].text(
                j, i, value, ha="center", va="center",
                color="white" if value > threshold else COLORS["navy"],
            )
    axes[0].set_xlabel("Grupo predicho")
    axes[0].set_ylabel("Grupo real")
    axes[0].set_title(
        f"Matriz de confusión (accuracy = {classifier['accuracy']:.0%}, "
        f"línea base = {classifier['baseline_accuracy']:.0%})",
        fontsize=10,
    )
    importances = classifier["importances"][::-1]
    axes[1].barh([FEATURE_LABELS[f] for f in importances.index], importances, color=COLORS["teal"])
    axes[1].set_title("Importancia de variables en el árbol")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "07_clasificador_perfiles.png", dpi=180)
    plt.close(fig)

    # 08. Recomendaciones por emoción
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.scatter(catalog["valence_mean"], catalog["arousal_mean"], s=10, color="#C9D1DA", alpha=0.6)
    examples = phase1["examples"]
    palette = list(COLORS.values()) + ["#4E8A3E", "#B5527D"]
    for color, (emotion, point) in zip(palette, EMOTIONS.items()):
        recs = examples[examples["emocion"] == emotion]
        ax.scatter(recs["valencia_percibida"], recs["activacion_percibida"], s=30, color=color)
        ax.scatter(*point, marker="*", s=260, color=color, edgecolor="black", linewidth=0.6)
        ax.annotate(emotion, point, xytext=(6, 6), textcoords="offset points", fontsize=10)
    ax.axvline(5, color="#657485", linestyle="--", linewidth=1)
    ax.axhline(5, color="#657485", linestyle="--", linewidth=1)
    ax.set(xlim=(1, 9), ylim=(1, 9), xlabel="Valencia", ylabel="Activación")
    ax.set_title("Prototipo: top-5 canciones para cada emoción (★ = emoción buscada)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "08_recomendaciones_por_emocion.png", dpi=180)
    plt.close(fig)

    # 09. Ruta emocional
    route = phase1["route"]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(catalog["valence_mean"], catalog["arousal_mean"], s=10, color="#C9D1DA", alpha=0.6)
    ax.plot(route["valencia_percibida"], route["activacion_percibida"], color=COLORS["navy"], linewidth=1.5)
    ax.scatter(
        route["valencia_percibida"], route["activacion_percibida"],
        c=route["paso"], cmap="coolwarm", s=90, zorder=3, edgecolor="black",
    )
    for _, row in route.iterrows():
        ax.annotate(
            f"{int(row['paso'])}. canción {int(row['song_id'])}",
            (row["valencia_percibida"], row["activacion_percibida"]),
            xytext=(7, -4), textcoords="offset points", fontsize=9,
        )
    ax.axvline(5, color="#657485", linestyle="--", linewidth=1)
    ax.axhline(5, color="#657485", linestyle="--", linewidth=1)
    ax.set(xlim=(1, 9), ylim=(1, 9), xlabel="Valencia", ylabel="Activación")
    ax.set_title("Ruta emocional: de «triste» a «alegre» en 6 canciones")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "09_ruta_emocional.png", dpi=180)
    plt.close(fig)


def run_phase1(zip_path: Path, output_dir: Path) -> dict[str, object]:
    analysis = compute_analysis(load_deam(zip_path))
    phase1 = compute_phase1(analysis)
    validate_phase1(phase1)
    save_phase1_tables(phase1, output_dir)
    save_phase1_plots(phase1, output_dir)
    return phase1


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Perfiles y recomendador emocional DEAM")
    parser.add_argument("--zip", dest="zip_path", type=Path, default=package_root / "DEAM_Annotations.zip")
    parser.add_argument("--output", dest="output_dir", type=Path, default=package_root / "outputs")
    parser.add_argument("--emocion", choices=sorted(EMOTIONS), help="Muestra recomendaciones para una emoción")
    parser.add_argument("--grupo", type=int, help="Personaliza según un grupo de oyentes")
    parser.add_argument("--k", type=int, default=5, help="Número de canciones a recomendar")
    args = parser.parse_args()

    phase1 = run_phase1(args.zip_path.resolve(), args.output_dir.resolve())
    metrics = phase1["metrics"]
    print("Validación Fase I completada")
    print(f"Evaluadores con perfil: {metrics['evaluadores_con_perfil']} de {metrics['evaluadores_totales']}")
    print(f"Grupos encontrados (k): {metrics['k_elegido']}  |  silhouette = {metrics['silhouette']:.3f}")
    print(phase1["cluster"]["summary"][["descripcion", "evaluadores"]].to_string())
    print(
        f"Clasificador: accuracy = {metrics['accuracy']:.1%} "
        f"(línea base {metrics['accuracy_linea_base']:.1%})"
    )

    if args.emocion:
        bias = (0.0, 0.0)
        title = f"\nRecomendaciones para «{args.emocion}»"
        if args.grupo is not None:
            if args.grupo not in phase1["biases"]:
                parser.error(f"--grupo debe estar entre 1 y {metrics['k_elegido']}")
            bias = phase1["biases"][args.grupo]
            title += f" personalizadas para el grupo {args.grupo}"
        print(title)
        print(recommend_emotion(phase1["catalog"], args.emocion, k=args.k, bias=bias).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
