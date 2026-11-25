"""Fase II: modelos de predicción emocional y evaluación del recomendador híbrido.

Ejecución completa (requiere haber corrido antes ``src/fase2_audio.py``):

    python src/fase2_modelos.py

Genera en ``outputs/`` las tablas ``fase2_*.csv``, ``metricas_fase2.json`` y
en ``models/`` el modelo AST final que usa la aplicación.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisis_deam import compute_analysis, load_deam
from fase2_audio import find_annotations_zip, load_ast, load_opensmile
from perfiles_recomendador import EMOTIONS, compute_phase1

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PACKAGE_ROOT / "outputs"
MODELS_DIR = PACKAGE_ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGETS = {"valencia": "valence_mean", "activacion": "arousal_mean"}
QUADRANTS = ["Negativa / baja", "Positiva / alta", "Positiva / baja", "Negativa / alta"]


# ---------------------------------------------------------------------------
# Datos y partición
# ---------------------------------------------------------------------------


def load_dataset() -> dict[str, object]:
    """Une etiquetas (valencia/activación por canción) con las dos representaciones."""
    analysis = compute_analysis(load_deam(find_annotations_zip()))
    static = analysis["static"].set_index("song_id")
    opensmile = load_opensmile()
    ast = load_ast()
    ids = static.index.intersection(opensmile.index).intersection(ast.index)
    static = static.loc[ids]
    representations = {
        "openSMILE (línea base)": opensmile.loc[ids],
        "AST (transfer learning)": ast.loc[ids],
        "openSMILE + AST": pd.concat([opensmile.loc[ids], ast.loc[ids]], axis=1),
    }
    train_ids, test_ids = train_test_split(
        ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=static["quadrant"]
    )
    return {
        "analysis": analysis,
        "static": static,
        "representations": representations,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }


def quadrant_of(valence, arousal) -> np.ndarray:
    valence, arousal = np.asarray(valence), np.asarray(arousal)
    return np.select(
        [
            (valence < 5) & (arousal < 5),
            (valence >= 5) & (arousal >= 5),
            (valence >= 5) & (arousal < 5),
            (valence < 5) & (arousal >= 5),
        ],
        QUADRANTS,
        default="Sin clasificar",
    )


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------


def candidate_models() -> dict[str, object]:
    return {
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 25))),
        "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1)),
        "Gradient boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def compare_models(data: dict[str, object]) -> tuple[pd.DataFrame, dict]:
    """Elige el modelo con validación cruzada en entrenamiento y lo mide en prueba.

    La partición de prueba (20 %) es la misma para todas las representaciones
    y solo se usa al final, como exige la regla del diseño experimental.
    """
    static = data["static"]
    train_ids, test_ids = data["train_ids"], data["test_ids"]
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, predictions = [], {}

    # Línea base trivial: predecir siempre el promedio del entrenamiento.
    for label, column in TARGETS.items():
        dummy = DummyRegressor().fit(np.zeros((len(train_ids), 1)), static.loc[train_ids, column])
        pred = dummy.predict(np.zeros((len(test_ids), 1)))
        rows.append({"representacion": "Promedio (sin audio)", "modelo": "Media", "dimension": label,
                     "RMSE_cv": np.nan, **regression_metrics(static.loc[test_ids, column], pred)})
        predictions[("Promedio (sin audio)", label)] = pred

    for rep_name, features in data["representations"].items():
        x_train, x_test = features.loc[train_ids], features.loc[test_ids]
        for label, column in TARGETS.items():
            y_train, y_test = static.loc[train_ids, column], static.loc[test_ids, column]
            for model_name, model in candidate_models().items():
                cv_pred = cross_val_predict(clone(model), x_train, y_train, cv=cv, n_jobs=-1)
                fitted = clone(model).fit(x_train, y_train)
                pred = fitted.predict(x_test)
                rows.append({
                    "representacion": rep_name, "modelo": model_name, "dimension": label,
                    "RMSE_cv": float(mean_squared_error(y_train, cv_pred) ** 0.5),
                    **regression_metrics(y_test, pred),
                })
                predictions[(rep_name, model_name, label)] = pred
    return pd.DataFrame(rows), predictions


def best_by_cv(results: pd.DataFrame) -> pd.DataFrame:
    """Para cada representación y dimensión, el modelo con menor RMSE en validación cruzada."""
    audio = results[results["representacion"] != "Promedio (sin audio)"]
    best = audio.loc[audio.groupby(["representacion", "dimension"])["RMSE_cv"].idxmin()]
    baseline = results[results["representacion"] == "Promedio (sin audio)"]
    return pd.concat([baseline, best], ignore_index=True)


# ---------------------------------------------------------------------------
# Evaluación del recomendador (sin usuarios reales)
# ---------------------------------------------------------------------------


def recommend_ids(coords: pd.DataFrame, target: tuple[float, float], k: int, penalty=None) -> np.ndarray:
    distance = np.hypot(coords["valencia"] - target[0], coords["activacion"] - target[1])
    score = distance if penalty is None else distance + 0.3 * penalty
    return score.nsmallest(k).index.to_numpy()


def evaluate_recommender(
    true_coords: pd.DataFrame, systems: dict[str, pd.DataFrame], k: int = 10, n_random: int = 200
) -> pd.DataFrame:
    """Mide, con las anotaciones humanas como verdad, qué tan buenas son las listas.

    - **Acierto de cuadrante@k:** % de canciones recomendadas cuyo cuadrante
      real coincide con el de la emoción pedida.
    - **Distancia real:** distancia media (anotada) entre canción y emoción.
    - **Diversidad:** distancia media entre pares de canciones de la lista.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    systems = {"Aleatorio": None, **systems}
    for name, coords in systems.items():
        hits, dists, divs = [], [], []
        for emotion, target in EMOTIONS.items():
            target_quadrant = quadrant_of(*target)
            repetitions = n_random if coords is None else 1
            for _ in range(repetitions):
                if coords is None:
                    ids = rng.choice(true_coords.index, size=k, replace=False)
                else:
                    ids = recommend_ids(coords, target, k)
                real = true_coords.loc[ids]
                hits.append(np.mean(quadrant_of(real["valencia"], real["activacion"]) == target_quadrant))
                dists.append(np.hypot(real["valencia"] - target[0], real["activacion"] - target[1]).mean())
                pts = real[["valencia", "activacion"]].to_numpy()
                pair = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
                divs.append(pair[np.triu_indices(k, 1)].mean())
        rows.append({
            "sistema": name,
            f"acierto_cuadrante@{k}": float(np.mean(hits)),
            "distancia_real_media": float(np.mean(dists)),
            "diversidad_lista": float(np.mean(divs)),
        })
    return pd.DataFrame(rows)


def personalization_effect(catalog: pd.DataFrame, biases: dict[int, tuple[float, float]], k: int = 10) -> pd.DataFrame:
    """Solapamiento de las listas entre perfiles: 0 % = listas totalmente distintas."""
    rows = []
    groups = sorted(biases)
    for emotion, target in EMOTIONS.items():
        lists = {}
        for g in groups:
            shifted = catalog.assign(
                valencia=catalog["valencia"] + biases[g][0],
                activacion=catalog["activacion"] + biases[g][1],
            )
            lists[g] = set(recommend_ids(shifted, target, k, penalty=catalog["incertidumbre"]))
        overlap = len(lists[groups[0]] & lists[groups[1]]) / k
        rows.append({"emocion": emotion, "solapamiento_listas": overlap})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------


def run() -> dict[str, object]:
    data = load_dataset()
    static = data["static"]
    results, _ = compare_models(data)
    best = best_by_cv(results)

    # Predicciones fuera de muestra (out-of-fold) para todo el catálogo, con el
    # mejor modelo de cada representación: simulan canciones sin anotar.
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = {}
    final_models = {}
    for rep_name, features in data["representations"].items():
        coords = {}
        for label, column in TARGETS.items():
            row = best[(best["representacion"] == rep_name) & (best["dimension"] == label)].iloc[0]
            model = candidate_models()[row["modelo"]]
            coords[label] = cross_val_predict(clone(model), features, static[column], cv=cv, n_jobs=-1)
            if rep_name == "AST (transfer learning)":
                final_models[label] = clone(model).fit(features, static[column])
        oof[rep_name] = pd.DataFrame(coords, index=static.index).clip(1, 9)

    true_coords = static[["valence_mean", "arousal_mean"]].set_axis(["valencia", "activacion"], axis=1)
    systems = {f"Predicho con {name}": coords for name, coords in oof.items()}
    systems["Anotaciones humanas (techo)"] = true_coords
    recommender_eval = evaluate_recommender(true_coords, systems)

    # Cuadrante predicho vs. real (exactitud de la clasificación emocional).
    quadrant_acc = {
        name: float(np.mean(quadrant_of(c["valencia"], c["activacion"]) == static["quadrant"].to_numpy()))
        for name, c in oof.items()
    }

    phase1 = compute_phase1(data["analysis"])
    catalog = true_coords.assign(incertidumbre=(static["valence_std"] + static["arousal_std"]) / 2)
    personalization = personalization_effect(catalog, phase1["biases"])

    # Guardar
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_DIR / "fase2_comparacion_modelos.csv", index=False)
    best.to_csv(OUTPUT_DIR / "fase2_mejores_modelos.csv", index=False)
    recommender_eval.to_csv(OUTPUT_DIR / "fase2_evaluacion_recomendador.csv", index=False)
    personalization.to_csv(OUTPUT_DIR / "fase2_efecto_personalizacion.csv", index=False)
    predicted = pd.concat({k: v for k, v in oof.items()}, axis=1)
    predicted.columns = [f"{rep}|{dim}" for rep, dim in predicted.columns]
    predicted.round(4).to_csv(OUTPUT_DIR / "fase2_predicciones_oof.csv")
    joblib.dump(
        {"modelos": final_models, "descripcion": "Regresores sobre embeddings AST (1536 valores)"},
        MODELS_DIR / "modelo_emocion_ast.joblib",
        compress=3,
    )

    metrics = {
        "canciones": int(len(static)),
        "entrenamiento": int(len(data["train_ids"])),
        "prueba": int(len(data["test_ids"])),
        "mejores_modelos_prueba": best.round(4).to_dict(orient="records"),
        "exactitud_cuadrante_oof": quadrant_acc,
        "evaluacion_recomendador": recommender_eval.round(4).to_dict(orient="records"),
        "solapamiento_medio_listas_entre_perfiles": float(personalization["solapamiento_listas"].mean()),
    }
    (OUTPUT_DIR / "metricas_fase2.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "data": data, "results": results, "best": best, "oof": oof,
        "recommender_eval": recommender_eval, "quadrant_acc": quadrant_acc,
        "personalization": personalization, "metrics": metrics, "phase1": phase1,
    }


if __name__ == "__main__":
    out = run()
    pd.set_option("display.width", 200)
    print(out["best"].round(3).to_string(index=False))
    print("\nExactitud de cuadrante (fuera de muestra):", {k: round(v, 3) for k, v in out["quadrant_acc"].items()})
    print(out["recommender_eval"].round(3).to_string(index=False))
    print("\nSolapamiento medio entre perfiles:", round(out["metrics"]["solapamiento_medio_listas_entre_perfiles"], 2))
