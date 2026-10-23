from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATIC_AVG_DIR = "annotations/annotations averaged per song/song_level/"
STATIC_RATER_DIR = "annotations/annotations per each rater/song_level/"
DYNAMIC_AVG_DIR = "annotations/annotations averaged per song/dynamic (per second annotations)/"

COLORS = {
    "navy": "#233A55",
    "blue": "#7398B8",
    "teal": "#5EAAA4",
    "mustard": "#D5A742",
    "coral": "#E97A62",
    "violet": "#8073A9",
}


def read_zip_csv(archive: ZipFile, member: str) -> pd.DataFrame:
    frame = pd.read_csv(archive.open(member), skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def load_deam(zip_path: Path) -> dict[str, object]:
    with ZipFile(zip_path) as archive:
        members = archive.namelist()
        csv_files = [name for name in members if name.lower().endswith(".csv")]

        static = pd.concat(
            [
                read_zip_csv(
                    archive,
                    STATIC_AVG_DIR + "static_annotations_averaged_songs_1_2000.csv",
                ),
                read_zip_csv(
                    archive,
                    STATIC_AVG_DIR + "static_annotations_averaged_songs_2000_2058.csv",
                ),
            ],
            ignore_index=True,
            sort=False,
        )

        rater_1 = read_zip_csv(
            archive,
            STATIC_RATER_DIR + "static_annotations_songs_1_2000.csv",
        ).rename(
            columns={
                "workerID": "worker_id",
                "SongId": "song_id",
                "Valence": "valence",
                "Arousal": "arousal",
            }
        )

        rater_2 = read_zip_csv(
            archive,
            STATIC_RATER_DIR + "static_annotations_songs_2000_2058.csv",
        ).rename(
            columns={
                "WorkerId": "worker_id",
                "SongId": "song_id",
                "Valence_Average": "valence",
                "Arousal_Average": "arousal",
            }
        )

        individual = pd.concat(
            [
                rater_1[["worker_id", "song_id", "valence", "arousal"]],
                rater_2[["worker_id", "song_id", "valence", "arousal"]],
            ],
            ignore_index=True,
        )

        dynamic_valence = read_zip_csv(archive, DYNAMIC_AVG_DIR + "valence.csv")
        dynamic_arousal = read_zip_csv(archive, DYNAMIC_AVG_DIR + "arousal.csv")

    static["song_id"] = pd.to_numeric(static["song_id"], errors="raise").astype(int)
    individual["song_id"] = pd.to_numeric(individual["song_id"], errors="raise").astype(int)
    dynamic_valence["song_id"] = pd.to_numeric(
        dynamic_valence["song_id"], errors="raise"
    ).astype(int)
    dynamic_arousal["song_id"] = pd.to_numeric(
        dynamic_arousal["song_id"], errors="raise"
    ).astype(int)

    return {
        "members": members,
        "csv_files": csv_files,
        "static": static,
        "individual": individual,
        "dynamic_valence": dynamic_valence,
        "dynamic_arousal": dynamic_arousal,
    }


def temporal_mean(frame: pd.DataFrame, output_name: str) -> pd.DataFrame:
    sample_columns = [column for column in frame.columns if column.startswith("sample_")]
    result = frame[["song_id"]].copy()
    result[output_name] = frame[sample_columns].mean(axis=1, skipna=True)
    return result


def compute_analysis(data: dict[str, object]) -> dict[str, object]:
    static = data["static"].copy()
    individual = data["individual"]
    dynamic_valence = data["dynamic_valence"]
    dynamic_arousal = data["dynamic_arousal"]

    sample_columns = [
        column for column in dynamic_valence.columns if column.startswith("sample_")
    ]
    sample_times_ms = sorted(
        int(re.search(r"(\d+)ms", column).group(1)) for column in sample_columns
    )
    sample_interval_ms = int(np.median(np.diff(sample_times_ms)))
    sampling_frequency_hz = 1000 / sample_interval_ms

    quadrant_names = [
        "Negativa / baja",
        "Positiva / alta",
        "Positiva / baja",
        "Negativa / alta",
    ]
    conditions = [
        (static["valence_mean"] < 5) & (static["arousal_mean"] < 5),
        (static["valence_mean"] >= 5) & (static["arousal_mean"] >= 5),
        (static["valence_mean"] >= 5) & (static["arousal_mean"] < 5),
        (static["valence_mean"] < 5) & (static["arousal_mean"] >= 5),
    ]
    static["quadrant"] = np.select(conditions, quadrant_names, default="Sin clasificar")
    quadrants = (
        static["quadrant"]
        .value_counts()
        .reindex(quadrant_names)
        .rename("songs")
        .to_frame()
    )
    quadrants["percentage"] = quadrants["songs"] / len(static) * 100

    dynamic_summary = temporal_mean(
        dynamic_valence, "dynamic_valence_mean"
    ).merge(
        temporal_mean(dynamic_arousal, "dynamic_arousal_mean"),
        on="song_id",
        how="inner",
    )
    comparison = static.merge(dynamic_summary, on="song_id", how="inner")
    comparison["static_valence_normalized"] = (
        comparison["valence_mean"] - 5
    ) / 4
    comparison["static_arousal_normalized"] = (
        comparison["arousal_mean"] - 5
    ) / 4

    metrics = {
        "songs": int(static["song_id"].nunique()),
        "individual_ratings": int(len(individual)),
        "unique_raters": int(individual["worker_id"].nunique()),
        "csv_files": int(len(data["csv_files"])),
        "median_raters_per_song": float(
            individual.groupby("song_id")["worker_id"].nunique().median()
        ),
        "sample_interval_ms": sample_interval_ms,
        "sampling_frequency_hz": float(sampling_frequency_hz),
        "valence_mean": float(static["valence_mean"].mean()),
        "valence_between_song_sd": float(static["valence_mean"].std()),
        "arousal_mean": float(static["arousal_mean"].mean()),
        "arousal_between_song_sd": float(static["arousal_mean"].std()),
        "valence_arousal_correlation": float(
            static["valence_mean"].corr(static["arousal_mean"])
        ),
        "mean_interrater_valence_sd": float(static["valence_std"].mean()),
        "mean_interrater_arousal_sd": float(static["arousal_std"].mean()),
        "valence_sd_over_1_5_pct": float((static["valence_std"] > 1.5).mean() * 100),
        "arousal_sd_over_1_5_pct": float((static["arousal_std"] > 1.5).mean() * 100),
        "static_dynamic_valence_correlation": float(
            comparison["valence_mean"].corr(comparison["dynamic_valence_mean"])
        ),
        "static_dynamic_arousal_correlation": float(
            comparison["arousal_mean"].corr(comparison["dynamic_arousal_mean"])
        ),
    }

    return {
        "static": static,
        "individual": individual,
        "comparison": comparison,
        "quadrants": quadrants,
        "metrics": metrics,
    }


def validate_results(analysis: dict[str, object]) -> None:
    metrics = analysis["metrics"]
    quadrants = analysis["quadrants"]
    expected_quadrants = {
        "Negativa / baja": 686,
        "Positiva / alta": 647,
        "Positiva / baja": 240,
        "Negativa / alta": 229,
    }

    assert metrics["songs"] == 1802
    assert metrics["individual_ratings"] == 17762
    assert metrics["unique_raters"] == 195
    assert metrics["csv_files"] == 3610
    assert metrics["sampling_frequency_hz"] == 2.0
    assert quadrants["songs"].to_dict() == expected_quadrants
    assert np.isclose(metrics["mean_interrater_valence_sd"], 1.502364, atol=1e-6)
    assert np.isclose(metrics["mean_interrater_arousal_sd"], 1.462719, atol=1e-6)
    assert np.isclose(
        metrics["static_dynamic_valence_correlation"], 0.734461, atol=1e-6
    )
    assert np.isclose(
        metrics["static_dynamic_arousal_correlation"], 0.676700, atol=1e-6
    )


def save_tables(analysis: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = analysis["metrics"]
    quadrants = analysis["quadrants"]

    results = pd.DataFrame(
        [
            ("Canciones o fragmentos", metrics["songs"], "conteo"),
            ("Valoraciones individuales", metrics["individual_ratings"], "conteo"),
            ("Evaluadores únicos", metrics["unique_raters"], "conteo"),
            ("Archivos CSV", metrics["csv_files"], "conteo"),
            ("Mediana de evaluadores por canción", metrics["median_raters_per_song"], "conteo"),
            ("Frecuencia dinámica", metrics["sampling_frequency_hz"], "Hz"),
            ("Valencia media", metrics["valence_mean"], "escala 1-9"),
            ("Activación media", metrics["arousal_mean"], "escala 1-9"),
            (
                "Correlación valencia-activación",
                metrics["valence_arousal_correlation"],
                "Pearson r",
            ),
            (
                "Desviación media entre evaluadores: valencia",
                metrics["mean_interrater_valence_sd"],
                "desviación",
            ),
            (
                "Desviación media entre evaluadores: activación",
                metrics["mean_interrater_arousal_sd"],
                "desviación",
            ),
            (
                "Canciones con desviación de valencia > 1.5",
                metrics["valence_sd_over_1_5_pct"],
                "%",
            ),
            (
                "Canciones con desviación de activación > 1.5",
                metrics["arousal_sd_over_1_5_pct"],
                "%",
            ),
            (
                "Correlación estática-dinámica: valencia",
                metrics["static_dynamic_valence_correlation"],
                "Pearson r",
            ),
            (
                "Correlación estática-dinámica: activación",
                metrics["static_dynamic_arousal_correlation"],
                "Pearson r",
            ),
        ],
        columns=["indicador", "resultado", "unidad"],
    )
    results.to_csv(output_dir / "resultados_presentacion.csv", index=False)
    quadrants.reset_index(names="quadrant").to_csv(
        output_dir / "distribucion_cuadrantes.csv", index=False
    )
    (output_dir / "metricas.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_plots(analysis: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    static = analysis["static"]
    comparison = analysis["comparison"]
    quadrants = analysis["quadrants"]
    metrics = analysis["metrics"]

    color_order = [
        COLORS["blue"],
        COLORS["teal"],
        COLORS["mustard"],
        COLORS["coral"],
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(quadrants.index, quadrants["songs"], color=color_order)
    ax.bar_label(bars, padding=3)
    ax.set_title("Distribución de canciones por cuadrante emocional")
    ax.set_ylabel("Número de canciones")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "01_distribucion_cuadrantes.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for quadrant, color in zip(quadrants.index, color_order):
        subset = static[static["quadrant"] == quadrant]
        ax.scatter(
            subset["valence_mean"],
            subset["arousal_mean"],
            s=25,
            alpha=0.55,
            label=quadrant,
            color=color,
        )
    ax.axvline(5, color="#657485", linestyle="--", linewidth=1)
    ax.axhline(5, color="#657485", linestyle="--", linewidth=1)
    ax.set(xlim=(1, 9), ylim=(1, 9), xlabel="Valencia media", ylabel="Activación media")
    ax.set_title("Espacio emocional de las canciones")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "02_espacio_valencia_activacion.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    dims = ["Valencia", "Activación"]
    mean_sd = [
        metrics["mean_interrater_valence_sd"],
        metrics["mean_interrater_arousal_sd"],
    ]
    over = [metrics["valence_sd_over_1_5_pct"], metrics["arousal_sd_over_1_5_pct"]]
    bars_1 = axes[0].bar(dims, mean_sd, color=[COLORS["coral"], COLORS["violet"]])
    bars_2 = axes[1].bar(dims, over, color=[COLORS["coral"], COLORS["violet"]])
    axes[0].bar_label(bars_1, fmt="%.2f", padding=3)
    axes[1].bar_label(bars_2, fmt="%.1f%%", padding=3)
    axes[0].set_title("Desviación media entre evaluadores")
    axes[1].set_title("Canciones con desviación superior a 1,5")
    axes[1].set_ylabel("Porcentaje")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "03_subjetividad_emocional.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [
        (
            "static_valence_normalized",
            "dynamic_valence_mean",
            COLORS["coral"],
            metrics["static_dynamic_valence_correlation"],
            "Valencia",
        ),
        (
            "static_arousal_normalized",
            "dynamic_arousal_mean",
            COLORS["violet"],
            metrics["static_dynamic_arousal_correlation"],
            "Activación",
        ),
    ]
    for axis, (x_col, y_col, color, correlation, title) in zip(axes, pairs):
        axis.scatter(comparison[x_col], comparison[y_col], s=20, alpha=0.3, color=color)
        slope, intercept = np.polyfit(comparison[x_col], comparison[y_col], 1)
        x_line = np.linspace(comparison[x_col].min(), comparison[x_col].max(), 100)
        axis.plot(x_line, slope * x_line + intercept, color=COLORS["navy"], linewidth=2)
        axis.set_title(f"{title}: r = {correlation:.3f}")
        axis.set_xlabel(f"{title} estática normalizada")
        axis.set_ylabel("Promedio de la señal dinámica")
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "04_consistencia_estatica_dinamica.png", dpi=180)
    plt.close(fig)


def run(zip_path: Path, output_dir: Path) -> dict[str, object]:
    data = load_deam(zip_path)
    analysis = compute_analysis(data)
    validate_results(analysis)
    save_tables(analysis, output_dir)
    save_plots(analysis, output_dir)
    return analysis


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Análisis reproducible de DEAM")
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=package_root / "DEAM_Annotations.zip",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        type=Path,
        default=package_root / "outputs",
    )
    args = parser.parse_args()

    analysis = run(args.zip_path.resolve(), args.output_dir.resolve())
    metrics = analysis["metrics"]
    print("Validación completada")
    print(f"Canciones: {metrics['songs']:,}")
    print(f"Valoraciones: {metrics['individual_ratings']:,}")
    print(f"Evaluadores: {metrics['unique_raters']:,}")
    print(f"Frecuencia dinámica: {metrics['sampling_frequency_hz']:.0f} Hz")
    print(
        "Correlaciones estática-dinámica: "
        f"valencia={metrics['static_dynamic_valence_correlation']:.3f}, "
        f"activación={metrics['static_dynamic_arousal_correlation']:.3f}"
    )


if __name__ == "__main__":
    main()
