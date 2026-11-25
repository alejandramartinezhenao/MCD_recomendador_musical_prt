"""Fase II: predicción de valencia y activación desde el audio.

Dos representaciones del audio, comparadas con la misma partición:

1. **Línea base acústica:** descriptores openSMILE que trae DEAM (`features/`),
   resumidos por canción con media y desviación de cada descriptor.
2. **Transfer learning:** embeddings del *Audio Spectrogram Transformer* (AST)
   preentrenado en AudioSet. El modelo se congela y solo se entrena un
   regresor ligero encima.

Uso desde la terminal (extrae y guarda las representaciones):

    python src/fase2_audio.py --opensmile
    python src/fase2_audio.py --ast

Rutas: el audio y los descriptores no van en el repositorio (pesan ~2 GB).
Se leen de la variable de entorno ``DEAM_DATA_DIR`` o, por defecto, de
``D:/D nuevo/Proyecto_Aplicado``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PACKAGE_ROOT / "outputs"
DATA_DIR = Path(os.getenv("DEAM_DATA_DIR", r"D:/D nuevo/Proyecto_Aplicado"))
AUDIO_DIR = DATA_DIR / "DEAM_audio" / "MEMD_audio"
FEATURES_DIR = DATA_DIR / "features" / "features"


def find_annotations_zip(root: Path = PACKAGE_ROOT) -> Path:
    """DEAM_Annotations.zip en la raíz (repositorio) o en data/raw (estructura Fase I)."""
    for candidate in (root / "DEAM_Annotations.zip", root / "data" / "raw" / "DEAM_Annotations.zip"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No se encontró DEAM_Annotations.zip")


OPENSMILE_PATH = OUTPUT_DIR / "fase2_opensmile_por_cancion.csv.gz"
AST_PATH = OUTPUT_DIR / "fase2_embeddings_ast.npz"

AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR = 16_000
SEGMENT_SECONDS = 10
MAX_SEGMENTS = 6
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Descriptores openSMILE (línea base)
# ---------------------------------------------------------------------------


def opensmile_song_vector(csv_path: Path) -> pd.Series:
    """Resume los descriptores por frame de una canción con su media y desviación."""
    frame = pd.read_csv(csv_path, sep=";")
    frame = frame.drop(columns=["frameTime"], errors="ignore")
    mean = frame.mean().add_suffix("__media")
    std = frame.std().add_suffix("__desv")
    return pd.concat([mean, std])


def build_opensmile_table(features_dir: Path = FEATURES_DIR) -> pd.DataFrame:
    rows = {}
    for path in sorted(features_dir.glob("*.csv"), key=lambda p: int(p.stem)):
        rows[int(path.stem)] = opensmile_song_vector(path)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "song_id"
    return table.astype("float32")


def load_opensmile() -> pd.DataFrame:
    return pd.read_csv(OPENSMILE_PATH, index_col="song_id")


# ---------------------------------------------------------------------------
# 2. Embeddings AST (transfer learning)
# ---------------------------------------------------------------------------


def load_audio(path, sr: int = AST_SR) -> np.ndarray:
    """Lee un archivo de audio (ruta u objeto tipo archivo) en mono a ``sr`` Hz."""
    import librosa
    import soundfile as sf

    signal, original_sr = sf.read(path, dtype="float32", always_2d=True)
    signal = signal.mean(axis=1)
    if original_sr != sr:
        signal = librosa.resample(signal, orig_sr=original_sr, target_sr=sr)
    return signal


def split_segments(signal: np.ndarray, sr: int = AST_SR) -> list[np.ndarray]:
    """Hasta ``MAX_SEGMENTS`` fragmentos de 10 s repartidos a lo largo del audio."""
    size = SEGMENT_SECONDS * sr
    if len(signal) <= size:
        return [signal]
    starts = np.linspace(0, len(signal) - size, num=min(MAX_SEGMENTS, len(signal) // size))
    return [signal[int(s) : int(s) + size] for s in starts]


class ASTEmbedder:
    """Envuelve el modelo AST congelado y devuelve un vector de 1.536 valores por canción."""

    def __init__(self, device: str | None = None) -> None:
        import torch
        from transformers import ASTFeatureExtractor, ASTModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL)
        self.model = ASTModel.from_pretrained(AST_MODEL, use_safetensors=True).to(self.device).eval()

    def embed_signal(self, signal: np.ndarray) -> np.ndarray:
        segments = split_segments(signal)
        inputs = self.extractor(segments, sampling_rate=AST_SR, return_tensors="pt")
        with self.torch.no_grad():
            pooled = self.model(inputs["input_values"].to(self.device)).pooler_output
        pooled = pooled.float().cpu().numpy()
        # media y desviación entre fragmentos: nivel emocional y cuánto cambia
        return np.concatenate([pooled.mean(axis=0), pooled.std(axis=0)]).astype("float32")

    def embed_file(self, path) -> np.ndarray:
        return self.embed_signal(load_audio(path))


def build_ast_embeddings(audio_dir: Path = AUDIO_DIR, output: Path = AST_PATH) -> None:
    embedder = ASTEmbedder()
    paths = sorted(audio_dir.glob("*.mp3"), key=lambda p: int(p.stem))
    ids, vectors = [], []
    start = time.time()
    for i, path in enumerate(paths, start=1):
        try:
            vectors.append(embedder.embed_file(path))
            ids.append(int(path.stem))
        except Exception as exc:  # un archivo dañado no detiene la extracción
            print(f"Omitido {path.name}: {type(exc).__name__}: {exc}")
        if i % 100 == 0:
            print(f"{i}/{len(paths)} canciones  ({time.time() - start:.0f} s)", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, song_id=np.array(ids), embedding=np.stack(vectors).astype("float16"))
    print(f"Guardado {output} con {len(ids)} canciones")


def load_ast() -> pd.DataFrame:
    data = np.load(AST_PATH)
    return pd.DataFrame(
        data["embedding"].astype("float32"),
        index=pd.Index(data["song_id"], name="song_id"),
        columns=[f"ast_{i}" for i in range(data["embedding"].shape[1])],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae representaciones de audio DEAM")
    parser.add_argument("--opensmile", action="store_true", help="Resume los descriptores openSMILE")
    parser.add_argument("--ast", action="store_true", help="Extrae embeddings AST (usa GPU si hay)")
    args = parser.parse_args()
    if args.opensmile:
        table = build_opensmile_table()
        OPENSMILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        table.round(5).to_csv(OPENSMILE_PATH)
        print(f"Guardado {OPENSMILE_PATH}: {table.shape}")
    if args.ast:
        build_ast_embeddings()


if __name__ == "__main__":
    main()
