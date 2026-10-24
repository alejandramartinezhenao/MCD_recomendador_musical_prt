# Análisis reproducible de DEAM

Este paquete contiene el material utilizado para obtener los resultados incluidos en la presentación del proyecto **Sistema recomendador musical basado en percepción emocional**.

## Contenido

- `analisis_deam_resultados_presentacion.ipynb`: notebook principal.
- `DEAM_Annotations.zip`: fuente de datos analizada.
- `src/analisis_deam.py`: funciones reutilizables y validaciones automáticas.
- `src/perfiles_recomendador.py`: Fase I (perfiles de anotación, K-Means, clasificador) y prototipo de recomendador emocional.
- `outputs/resultados_presentacion.csv`: indicadores utilizados en las diapositivas.
- `outputs/distribucion_cuadrantes.csv`: conteos y porcentajes por cuadrante emocional.
- `outputs/metricas.json`: resultados en formato estructurado.
- `outputs/*.png`: gráficas generadas por el análisis.
- `requirements.txt`: dependencias de Python.

## Requisitos

- Python 3.10 o superior.
- Aproximadamente 150 MB libres para instalar dependencias y ejecutar el análisis.

## Instalación

Desde la carpeta descomprimida:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

En macOS o Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Abra `analisis_deam_resultados_presentacion.ipynb` y seleccione **Run All**.

## Ejecución sin Jupyter

El análisis también se puede reproducir desde la terminal:

```bash
python src/analisis_deam.py
```

El script vuelve a generar todos los archivos de la carpeta `outputs/`.

### Fase I y prototipo de recomendador

```bash
python src/perfiles_recomendador.py
python src/perfiles_recomendador.py --emocion triste
python src/perfiles_recomendador.py --emocion alegre --grupo 2
```

Emociones disponibles: alegre, energico, relajado, tranquilo, triste, melancolico, tenso, enojado.

Genera en `outputs/`: `perfiles_evaluadores.csv`, `resumen_grupos.csv`, `seleccion_k.csv`, `reglas_clasificador.txt`, `recomendaciones_por_emocion.csv`, `ruta_emocional_triste_alegre.csv`, `metricas_fase1.json` y las gráficas `05` a `09`.

## Validaciones incluidas

El código verifica automáticamente los principales resultados:

- 1.802 canciones o fragmentos;
- 17.762 valoraciones individuales;
- 195 evaluadores únicos;
- 3.610 archivos CSV;
- frecuencia dinámica de 2 Hz;
- cuadrantes: 686, 647, 240 y 229 canciones;
- correlación estática-dinámica de valencia cercana a 0,734;
- correlación estática-dinámica de activación cercana a 0,677.

Fase I:

- 116 evaluadores con al menos 10 canciones;
- número de grupos elegido entre 2 y 6 por silhouette;
- precisión del clasificador superior a la línea base.

Si la fuente cambia y alguno de estos valores deja de coincidir, la celda de validación y el script producirán un error explícito.

## Alcance de los datos

El archivo incluye anotaciones emocionales estáticas y dinámicas. No contiene los archivos de audio ni interacciones históricas usuario-canción. Por esta razón, permite reproducir el análisis emocional de la presentación, pero no entrenar todavía el componente de transfer learning ni un recomendador colaborativo.

