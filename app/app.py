"""Demo del recomendador musical basado en percepción emocional.

Ejecutar desde la carpeta del repositorio:

    streamlit run app/app.py

Si el audio de DEAM está disponible (variable DEAM_DATA_DIR o la ruta por
defecto), la app permite escuchar cada canción recomendada.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analisis_deam import compute_analysis, load_deam  # noqa: E402
from fase2_audio import AUDIO_DIR, find_annotations_zip  # noqa: E402
from perfiles_recomendador import (  # noqa: E402
    EMOTIONS,
    PROFILE_FEATURES,
    compute_phase1,
    emotional_route,
    recommend,
)

st.set_page_config(page_title="Recomendador emocional DEAM", page_icon="🎵", layout="wide")

EMOTION_LABELS = {
    "alegre": "😄 Alegre", "energico": "⚡ Enérgico", "relajado": "😌 Relajado",
    "tranquilo": "🌙 Tranquilo", "triste": "😢 Triste", "melancolico": "🌧️ Melancólico",
    "tenso": "😬 Tenso", "enojado": "😠 Enojado",
}
GROUP_NAMES = {
    0: "Sin personalizar",
    1: "Perfil 1 · percibe más positividad y energía",
    2: "Perfil 2 · percibe más negatividad y calma",
}
QUIZ_SIZE = 10


# ---------------------------------------------------------------------------
# Carga (se hace una sola vez)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Cargando DEAM y perfiles de la Fase I…")
def load_project():
    analysis = compute_analysis(load_deam(find_annotations_zip()))
    phase1 = compute_phase1(analysis)
    return analysis, phase1


@st.cache_resource(show_spinner="Cargando el modelo de audio (AST)…")
def load_audio_model():
    import joblib

    from fase2_audio import ASTEmbedder

    return ASTEmbedder(), joblib.load(ROOT / "models" / "modelo_emocion_ast.joblib")["modelos"]


def audio_path(song_id: int) -> Path | None:
    path = AUDIO_DIR / f"{int(song_id)}.mp3"
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Componentes visuales
# ---------------------------------------------------------------------------


def emotion_map(catalog: pd.DataFrame, recs: pd.DataFrame | None = None, target=None,
                route: pd.DataFrame | None = None, extra=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=catalog["valence_mean"], y=catalog["arousal_mean"], mode="markers",
        marker=dict(size=5, color="#C9D1DA"), name="Catálogo DEAM",
        text=[f"Canción {s}" for s in catalog["song_id"]], hoverinfo="text",
    ))
    if recs is not None and len(recs):
        fig.add_trace(go.Scatter(
            x=recs["valencia_percibida"], y=recs["activacion_percibida"], mode="markers+text",
            marker=dict(size=13, color="#E97A62", line=dict(color="white", width=1)),
            text=[str(i + 1) for i in range(len(recs))], textposition="top center",
            name="Recomendadas", hovertext=[f"Canción {s}" for s in recs["song_id"]], hoverinfo="text",
        ))
    if route is not None:
        fig.add_trace(go.Scatter(
            x=route["valencia_percibida"], y=route["activacion_percibida"], mode="lines+markers+text",
            line=dict(color="#233A55", width=2), marker=dict(size=12, color="#5EAAA4"),
            text=[f"{p}" for p in route["paso"]], textposition="top center", name="Ruta",
        ))
    if target is not None:
        fig.add_trace(go.Scatter(
            x=[target[0]], y=[target[1]], mode="markers",
            marker=dict(symbol="star", size=24, color="#D5A742", line=dict(color="black", width=1)),
            name="Emoción buscada",
        ))
    if extra is not None:
        fig.add_trace(go.Scatter(
            x=[extra[0]], y=[extra[1]], mode="markers",
            marker=dict(symbol="diamond", size=20, color="#8073A9", line=dict(color="black", width=1)),
            name="Tu canción",
        ))
    for value in (5,):
        fig.add_hline(y=value, line_dash="dot", line_color="#8592A3")
        fig.add_vline(x=value, line_dash="dot", line_color="#8592A3")
    for x, y, txt in [(2.2, 8.7, "Tensa / enojo"), (7.8, 8.7, "Alegre / enérgica"),
                      (2.2, 1.3, "Triste / calmada"), (7.8, 1.3, "Relajada / serena")]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False, font=dict(color="#657485", size=12))
    fig.update_layout(
        xaxis=dict(title="Valencia (negativa → positiva)", range=[1, 9]),
        yaxis=dict(title="Activación (calma → energía)", range=[1, 9]),
        height=520, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.15), plot_bgcolor="white",
    )
    return fig


def show_list(recs: pd.DataFrame) -> None:
    table = recs.rename(columns={
        "song_id": "Canción", "valencia_percibida": "Valencia", "activacion_percibida": "Activación",
        "quadrant": "Cuadrante", "distancia": "Distancia", "incertidumbre": "Desacuerdo",
        "puntaje": "Puntaje",
    })
    table.index = range(1, len(table) + 1)
    st.dataframe(table.round(2), width="stretch")
    playable = [s for s in recs["song_id"] if audio_path(s)]
    if playable:
        with st.expander("🎧 Escuchar las canciones recomendadas", expanded=True):
            cols = st.columns(2)
            for i, song in enumerate(recs["song_id"]):
                path = audio_path(song)
                if path:
                    with cols[i % 2]:
                        st.caption(f"{i + 1}. Canción {int(song)}")
                        st.audio(str(path))
    else:
        st.caption("Audio no disponible en este equipo (configura DEAM_DATA_DIR para escuchar).")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

analysis, phase1 = load_project()
catalog = phase1["catalog"]
biases = phase1["biases"]

st.title("🎵 Recomendador musical basado en percepción emocional")
st.caption("Proyecto Aplicado I · Darío Alexander Ojeda Muñoz · Alejandra Martínez Henao · Dataset DEAM")

tab_rec, tab_perfil, tab_ruta, tab_audio, tab_res = st.tabs([
    "🎯 Recomendar por emoción", "🧭 Descubre tu perfil", "🛤️ Ruta emocional",
    "🎙️ Analiza una canción", "📊 Resultados",
])

with tab_rec:
    left, right = st.columns([1, 2])
    with left:
        emotion = st.radio("¿Cómo te quieres sentir?", list(EMOTIONS), format_func=EMOTION_LABELS.get, horizontal=True)
        default_group = st.session_state.get("grupo_usuario", 0)
        group = st.selectbox("Perfil de percepción", list(GROUP_NAMES), index=default_group, format_func=GROUP_NAMES.get)
        k = st.slider("Número de canciones", 3, 10, 5)
        with st.expander("Ajuste fino (opcional)"):
            weight = st.slider("Peso del desacuerdo entre evaluadores", 0.0, 1.0, 0.3, 0.05,
                               help="Más alto = prefiere canciones en las que la gente coincide.")
        st.markdown("**Cómo funciona:** puntaje = distancia a la emoción + peso × desacuerdo. "
                    "El perfil corrige cómo percibe ese grupo cada canción.")
    target = EMOTIONS[emotion]
    bias = biases.get(group, (0.0, 0.0))
    recs = recommend(catalog, *target, k=k, bias=bias, uncertainty_weight=weight)
    with right:
        st.plotly_chart(emotion_map(catalog, recs, target), width="stretch")
    show_list(recs)

with tab_perfil:
    st.markdown(
        f"Escucha y califica **{QUIZ_SIZE} canciones**. Comparamos tus respuestas con el consenso de los "
        "evaluadores de DEAM y el **árbol de decisión de la Fase I** te asigna un perfil."
    )
    quiz_pool = catalog.nsmallest(400, "incertidumbre")
    quiz = (quiz_pool.groupby("quadrant", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), 3), random_state=7))
            .head(QUIZ_SIZE).reset_index(drop=True))
    answers = []
    with st.form("quiz"):
        cols = st.columns(2)
        for i, row in quiz.iterrows():
            with cols[i % 2]:
                st.markdown(f"**Canción {int(row.song_id)}**")
                path = audio_path(row.song_id)
                if path:
                    st.audio(str(path))
                v = st.slider("¿Qué tan positiva? (1 = muy negativa, 9 = muy positiva)", 1, 9, 5, key=f"v{i}")
                a = st.slider("¿Qué tan enérgica? (1 = muy calmada, 9 = muy enérgica)", 1, 9, 5, key=f"a{i}")
                answers.append((v, a))
        sent = st.form_submit_button("Calcular mi perfil", type="primary")
    if sent:
        ans = np.array(answers, dtype=float)
        dv, da = ans[:, 0] - quiz["valence_mean"], ans[:, 1] - quiz["arousal_mean"]
        features = pd.DataFrame([{
            "valencia_media": ans[:, 0].mean(), "activacion_media": ans[:, 1].mean(),
            "valencia_sd": ans[:, 0].std(ddof=1), "activacion_sd": ans[:, 1].std(ddof=1),
            "sesgo_valencia": dv.mean(), "sesgo_activacion": da.mean(),
            "discrepancia": ((dv.abs() + da.abs()) / 2).mean(),
        }])[PROFILE_FEATURES]
        user_group = int(phase1["classifier"]["model"].predict(features)[0])
        st.session_state["grupo_usuario"] = user_group
        st.success(f"Tu perfil: **{GROUP_NAMES[user_group]}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tu sesgo en valencia", f"{dv.mean():+.2f}")
        c2.metric("Tu sesgo en activación", f"{da.mean():+.2f}")
        c3.metric("Distancia al consenso", f"{features['discrepancia'].iloc[0]:.2f}")
        st.info("Este perfil ya quedó seleccionado en la pestaña **Recomendar por emoción**.")

with tab_ruta:
    c1, c2, c3 = st.columns(3)
    start = c1.selectbox("Estado actual", list(EMOTIONS), index=list(EMOTIONS).index("triste"), format_func=EMOTION_LABELS.get)
    end = c2.selectbox("Estado deseado", list(EMOTIONS), index=list(EMOTIONS).index("alegre"), format_func=EMOTION_LABELS.get)
    steps = c3.slider("Canciones en la ruta", 3, 10, 6)
    st.caption("Principio iso de la musicoterapia: empezar donde está el oyente y avanzar poco a poco.")
    if start == end:
        st.warning("Elige dos emociones distintas.")
    else:
        route = emotional_route(catalog, start, end, steps=steps)
        st.plotly_chart(emotion_map(catalog, route=route, target=EMOTIONS[end]), width="stretch")
        show_list(route[["song_id", "valencia_percibida", "activacion_percibida", "quadrant", "distancia", "incertidumbre", "puntaje"]])

with tab_audio:
    st.markdown(
        "Sube una canción (mp3 o wav). El modelo **AST preentrenado** (transfer learning) extrae su "
        "representación y un regresor entrenado con DEAM predice su **valencia y activación**. "
        "Luego te recomendamos canciones con una emoción parecida."
    )
    source = st.radio("Origen", ["Subir archivo", "Elegir una canción de DEAM"], horizontal=True)
    audio_bytes, label, true_point = None, None, None
    if source == "Subir archivo":
        upload = st.file_uploader("Archivo de audio", type=["mp3", "wav", "flac", "ogg"])
        if upload:
            audio_bytes, label = upload, upload.name
    else:
        song = st.number_input("ID de canción", min_value=int(catalog.song_id.min()),
                               max_value=int(catalog.song_id.max()), value=10)
        path = audio_path(song)
        if path is None:
            st.warning("Esa canción no está disponible en este equipo.")
        else:
            audio_bytes, label = str(path), f"Canción {song}"
            row = catalog[catalog.song_id == song]
            if len(row):
                true_point = (float(row.valence_mean.iloc[0]), float(row.arousal_mean.iloc[0]))
    if audio_bytes is not None and st.button("Analizar emoción", type="primary"):
        embedder, models = load_audio_model()
        with st.spinner("Escuchando la canción…"):
            vector = embedder.embed_file(audio_bytes).reshape(1, -1)
            feats = pd.DataFrame(vector, columns=[f"ast_{i}" for i in range(vector.shape[1])])
            val = float(np.clip(models["valencia"].predict(feats)[0], 1, 9))
            aro = float(np.clip(models["activacion"].predict(feats)[0], 1, 9))
        st.audio(audio_bytes)
        c1, c2, c3 = st.columns(3)
        c1.metric("Valencia predicha", f"{val:.1f} / 9")
        c2.metric("Activación predicha", f"{aro:.1f} / 9")
        if true_point:
            c3.metric("Anotación humana (DEAM)", f"V {true_point[0]:.1f} · A {true_point[1]:.1f}")
            st.caption("Nota: las canciones de DEAM se usaron para entrenar el modelo final, así que aquí el acierto "
                       "es optimista. El error real con canciones nuevas es ≈ 0,8 puntos en valencia y ≈ 0,7 en activación.")
        exclude = {int(song)} if source != "Subir archivo" else None
        recs = recommend(catalog, val, aro, k=5, exclude=exclude)
        st.plotly_chart(emotion_map(catalog, recs, target=true_point, extra=(val, aro)), width="stretch")
        st.markdown(f"**Canciones con una emoción parecida a {label}:**")
        show_list(recs)

with tab_res:
    m = analysis["metrics"]
    m1 = phase1["metrics"]
    st.subheader("Fase I · Perfiles de percepción")
    c = st.columns(4)
    c[0].metric("Canciones", f"{m['songs']:,}".replace(",", "."))
    c[1].metric("Evaluadores con perfil", m1["evaluadores_con_perfil"])
    c[2].metric("Perfiles (K-Means)", m1["k_elegido"], f"silhouette {m1['silhouette']:.2f}", delta_color="off")
    c[3].metric("Acierto del árbol", f"{m1['accuracy']:.1%}", f"línea base {m1['accuracy_linea_base']:.1%}", delta_color="off")

    metrics_path = ROOT / "outputs" / "metricas_fase2.json"
    if metrics_path.exists():
        st.subheader("Fase II · Emoción desde el audio y recomendación")
        best = pd.read_csv(ROOT / "outputs" / "fase2_mejores_modelos.csv")
        pivot = best.pivot_table(index=["representacion", "modelo"], columns="dimension", values=["R2", "RMSE"])
        pivot.columns = [f"{a} {b}" for a, b in pivot.columns]
        st.dataframe(pivot.round(3), width="stretch")
        rec_eval = pd.read_csv(ROOT / "outputs" / "fase2_evaluacion_recomendador.csv").set_index("sistema")
        st.dataframe(rec_eval.drop(columns="cobertura_catalogo", errors="ignore").round(3), width="stretch")
        m2 = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.caption(
            f"Partición estratificada: {m2['entrenamiento']} canciones de entrenamiento y {m2['prueba']} de prueba. "
            "La evaluación del recomendador usa las anotaciones humanas como verdad."
        )
