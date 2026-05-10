# -*- coding: utf-8 -*-
"""Analisis de presencia de estres a partir de un video.

Este script esta enfocado solo en video:
1. Carga un modelo ya entrenado.
2. Lee una grabacion.
3. Detecta rostros en el video.
4. Predice si hay presencia de estres.
5. Agrupa detecciones por persona.
6. Guarda CSV, resumenes y rostros extraidos.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Any

try:
    import pandas as pd
    import numpy as np
    import tensorflow as tf
except ModuleNotFoundError as exc:
    missing_module = exc.name or "una dependencia requerida"
    raise SystemExit(
        "Falta instalar dependencias para ejecutar este script. "
        f"Modulo faltante: '{missing_module}'.\n"
        "Instala todo con:\n"
        "python3 -m pip install -r requirements.txt"
    ) from exc


IMG_SIZE = 48
CHANNELS = 1
MIN_CONFIDENCE = 25.0
MIN_FACE_STD = 18.0
MIN_FACE_MEAN = 20.0
MAX_FACE_MEAN = 235.0
STRESS_LABELS = (1, 0)
STATS_INTERVAL_SECONDS = 10
STRESS_BY_MODEL_OUTPUT = (1, 1, 1, 0, 1, 0, 0)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def importar_cv2() -> Any:
    """Importa OpenCV solo cuando se necesita."""
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV no esta instalado. Instala 'python3-opencv' o 'opencv-python'."
        ) from exc
    return cv2


def asegurar_directorio(path: Path | None) -> None:
    """Crea una carpeta si la ruta existe."""
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


def guardar_csv(df: pd.DataFrame, output_path: Path | None) -> None:
    """Guarda un DataFrame si se definio una ruta."""
    if output_path is None:
        return
    asegurar_directorio(output_path.parent)
    df.to_csv(output_path, index=False)
    print(f"\nArchivo guardado en: {output_path}")


def obtener_videos(video_paths: list[Path]) -> list[Path]:
    """Obtiene uno o varios videos desde un archivo o carpeta."""
    videos: list[Path] = []
    for video_path in video_paths:
        if video_path.is_file():
            videos.append(video_path)
            continue
        if video_path.is_dir():
            videos.extend(
                sorted(
                    path
                    for path in video_path.iterdir()
                    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
                )
            )
            continue
        raise FileNotFoundError(f"No existe la ruta indicada: {video_path}")
    return sorted(dict.fromkeys(videos))


def porcentaje_serie(serie: pd.Series, valor: int) -> float:
    """Calcula el porcentaje de un valor dentro de una serie."""
    return round(float((serie == valor).mean() * 100), 2)


def nombre_columna_estres(resultado: int) -> str:
    """Convierte un resultado en sufijo de columna."""
    return "estres" if resultado == 1 else "sin_estres"


def resultado_dominante_binario(serie: pd.Series) -> int:
    """Devuelve 1 si al menos la mitad de detecciones indica estres."""
    return int(float(serie.mean()) >= 0.5)


def convertir_salida_a_estres(class_id: int, output_size: int) -> int:
    """Convierte la salida del modelo a 1 con estres o 0 sin estres."""
    if output_size == len(STRESS_LABELS):
        return STRESS_LABELS[class_id]
    if output_size == len(STRESS_BY_MODEL_OUTPUT):
        return STRESS_BY_MODEL_OUTPUT[class_id]
    raise ValueError(
        "El modelo debe devolver 2 clases de estres o 7 clases compatibles "
        f"con el modelo actual. Salidas recibidas: {output_size}"
    )


def normalizar_gris(image: np.ndarray) -> np.ndarray:
    """Normaliza una imagen en grises para el modelo."""
    image = image.astype(np.float32) / 255.0
    return image.reshape(1, IMG_SIZE, IMG_SIZE, CHANNELS)


def predecir_estres(
    model: tf.keras.Model, gray_face: np.ndarray, cv2: Any
) -> tuple[int, float]:
    """Predice si un rostro presenta estres."""
    rostro = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
    pred = model.predict(normalizar_gris(rostro), verbose=0)[0]
    class_id = int(np.argmax(pred))
    confianza = float(np.max(pred) * 100)
    return convertir_salida_a_estres(class_id, len(pred)), confianza


def cargar_detector_rostros() -> tuple[Any, Any, list[Any]]:
    """Carga detectores Haar Cascade de rostro y ojos."""
    cv2 = importar_cv2()
    face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    eyes_paths = [
        cv2.data.haarcascades + "haarcascade_eye.xml",
        cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml",
    ]

    detector = cv2.CascadeClassifier(face_path)
    eye_detectors = [cv2.CascadeClassifier(path) for path in eyes_paths]

    if detector.empty():
        raise RuntimeError("No se pudo cargar el detector de rostros de OpenCV.")
    if any(eye_detector.empty() for eye_detector in eye_detectors):
        raise RuntimeError("No se pudo cargar el detector de ojos de OpenCV.")
    return cv2, detector, eye_detectors


def detectar_rostros(gray_image: np.ndarray, detector: Any) -> list[tuple[int, int, int, int]]:
    """Detecta rostros en una imagen gris."""
    faces = detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(24, 24),
    )
    return [tuple(map(int, face_box)) for face_box in faces]


def contar_ojos(upper_face: np.ndarray, eye_detectors: list[Any]) -> int:
    """Cuenta posibles ojos dentro de la zona superior del rostro."""
    eye_boxes: list[tuple[int, int, int, int]] = []

    for eye_detector in eye_detectors:
        eyes = eye_detector.detectMultiScale(
            upper_face,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(6, 6),
        )
        eye_boxes.extend(tuple(map(int, eye_box)) for eye_box in eyes)

    if not eye_boxes:
        return 0

    centers: set[tuple[int, int]] = set()
    for x, y, w, h in eye_boxes:
        centers.add((int((x + w / 2) // 5), int((y + h / 2) // 5)))
    return len(centers)


def es_rostro_valido(
    rostro: np.ndarray,
    face_box: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
    eye_detectors: list[Any],
    strict_face_check: bool,
) -> bool:
    """Filtra recortes que probablemente no sean rostros utiles."""
    x, y, w, h = face_box
    frame_h, frame_w = frame_shape

    min_lado = max(28, int(min(frame_h, frame_w) * 0.025))
    if w < min_lado or h < min_lado:
        return False

    aspect_ratio = w / h if h else 0
    if aspect_ratio < 0.6 or aspect_ratio > 1.8:
        return False

    if rostro.size == 0:
        return False

    mean_value = float(np.mean(rostro))
    std_value = float(np.std(rostro))
    if mean_value < MIN_FACE_MEAN or mean_value > MAX_FACE_MEAN:
        return False
    if std_value < MIN_FACE_STD:
        return False

    if x <= 1 or y <= 1 or x + w >= frame_w - 1 or y + h >= frame_h - 1:
        return False

    if strict_face_check:
        upper_face = rostro[: max(1, int(rostro.shape[0] * 0.7)), :]
        if contar_ojos(upper_face, eye_detectors) < 1:
            return False

    return True


def centro_caja(face_box: tuple[int, int, int, int]) -> tuple[float, float]:
    """Calcula el centro de una caja."""
    x, y, w, h = face_box
    return x + (w / 2), y + (h / 2)


def asignar_persona_id(
    face_box: tuple[int, int, int, int],
    trackers: dict[int, dict[str, float]],
    siguiente_id: int,
    frame_idx: int,
    max_distancia: float,
    max_salto_frames: int,
    ids_usados_frame: set[int] | None = None,
) -> tuple[int, int]:
    """Asigna un identificador simple por cercania entre frames."""
    cx, cy = centro_caja(face_box)
    mejor_id = None
    mejor_distancia = None
    if ids_usados_frame is None:
        ids_usados_frame = set()

    for persona_id, data in trackers.items():
        if persona_id in ids_usados_frame:
            continue
        if frame_idx - int(data["ultimo_frame"]) > max_salto_frames:
            continue

        distancia = ((cx - data["cx"]) ** 2 + (cy - data["cy"]) ** 2) ** 0.5
        if distancia <= max_distancia and (
            mejor_distancia is None or distancia < mejor_distancia
        ):
            mejor_distancia = distancia
            mejor_id = persona_id

    if mejor_id is None:
        mejor_id = siguiente_id
        siguiente_id += 1

    trackers[mejor_id] = {"cx": cx, "cy": cy, "ultimo_frame": float(frame_idx)}
    ids_usados_frame.add(mejor_id)
    return mejor_id, siguiente_id


def guardar_rostro_video(
    cv2: Any,
    rostro: np.ndarray,
    faces_dir: Path,
    persona_id: int,
    frame_idx: int,
    segundo: float,
) -> str:
    """Guarda el recorte facial del video por persona."""
    persona_dir = faces_dir / f"persona_{persona_id:03d}"
    asegurar_directorio(persona_dir)
    filename = f"frame_{frame_idx:06d}_seg_{segundo:08.2f}.png"
    output_path = persona_dir / filename
    if not cv2.imwrite(str(output_path), rostro):
        raise RuntimeError(f"No se pudo guardar el rostro extraido: {output_path}")
    return str(output_path)


def resumir_video_por_persona(resultados_df: pd.DataFrame) -> pd.DataFrame:
    """Genera un resumen por persona detectada."""
    resumen: list[dict[str, object]] = []

    for persona_id, grupo in resultados_df.groupby("persona_id"):
        rostro_representativo = grupo.sort_values(by="confianza", ascending=False).iloc[0]
        estres = grupo["estres_predicho"]
        resumen.append(
            {
                "persona_id": persona_id,
                "rostro_representativo": rostro_representativo["rostro_path"],
                "detecciones": len(grupo),
                "inicio_segundo": round(float(grupo["segundo"].min()), 2),
                "fin_segundo": round(float(grupo["segundo"].max()), 2),
                "estres_dominante": resultado_dominante_binario(estres),
                "confianza_media": round(float(grupo["confianza"].mean()), 2),
                **{
                    f"porcentaje_{nombre_columna_estres(nivel)}": porcentaje_serie(estres, nivel)
                    for nivel in STRESS_LABELS
                },
            }
        )

    return pd.DataFrame(resumen).sort_values(
        by=["porcentaje_estres", "detecciones"],
        ascending=[False, False],
    )


def calcular_estadistica_general(resultados_df: pd.DataFrame) -> pd.DataFrame:
    """Genera una tabla estadistica general de los rostros analizados."""
    total = len(resultados_df)
    filas = [
        {
            "estres_predicho": nivel,
            "descripcion": "CON_ESTRES" if nivel == 1 else "SIN_ESTRES",
            "imagenes_analizadas": int((resultados_df["estres_predicho"] == nivel).sum()),
            "porcentaje": porcentaje_serie(resultados_df["estres_predicho"], nivel),
        }
        for nivel in STRESS_LABELS
    ]
    filas.append({
        "estres_predicho": "TOTAL",
        "descripcion": "TOTAL",
        "imagenes_analizadas": total,
        "porcentaje": 100.0,
    })
    return pd.DataFrame(filas)


def calcular_estadistica_por_intervalo(
    resultados_df: pd.DataFrame,
    interval_seconds: int = STATS_INTERVAL_SECONDS,
) -> pd.DataFrame:
    """Agrupa las detecciones por intervalos de tiempo del video."""
    if interval_seconds <= 0:
        raise ValueError("El intervalo estadistico debe ser mayor que 0.")

    df = resultados_df.copy()
    df["intervalo_inicio_seg"] = (
        df["segundo"].astype(float) // interval_seconds * interval_seconds
    ).astype(int)
    df["intervalo_fin_seg"] = df["intervalo_inicio_seg"] + interval_seconds

    filas = []
    for (inicio, fin), grupo in df.groupby(["intervalo_inicio_seg", "intervalo_fin_seg"]):
        total = len(grupo)
        estres = grupo["estres_predicho"]
        filas.append({
            "intervalo_inicio_seg": int(inicio),
            "intervalo_fin_seg": int(fin),
            "imagenes_analizadas": total,
            "personas_detectadas": int(grupo["persona_id"].nunique()),
            "confianza_media": round(float(grupo["confianza"].mean()), 2),
            **{
                nombre: valor
                for nivel in STRESS_LABELS
                for nombre, valor in (
                    (f"cantidad_{nombre_columna_estres(nivel)}", int((estres == nivel).sum())),
                    (f"porcentaje_{nombre_columna_estres(nivel)}", porcentaje_serie(estres, nivel)),
                )
            },
        })

    return pd.DataFrame(filas).sort_values(by="intervalo_inicio_seg")


def crear_tabla_por_imagen(resultados_df: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabla simple con una fila por imagen analizada."""
    tabla = resultados_df[
        [
            "video",
            "frame",
            "segundo",
            "persona_id",
            "rostro_path",
            "estres_predicho",
            "confianza",
        ]
    ].copy()
    tabla["descripcion"] = tabla["estres_predicho"].map({1: "CON_ESTRES", 0: "SIN_ESTRES"})
    return tabla.sort_values(by=["persona_id", "frame", "segundo"]).reset_index(drop=True)


def guardar_tablas_estadisticas(
    resultados_df: pd.DataFrame,
    resumen_df: pd.DataFrame,
    output_csv: Path,
    stats_interval_seconds: int = STATS_INTERVAL_SECONDS,
) -> None:
    """Guarda tablas estadisticas del analisis de imagenes del video."""
    stats_dir = output_csv.parent / f"{output_csv.stem}_tablas_estadisticas"
    asegurar_directorio(stats_dir)

    estadistica_general = calcular_estadistica_general(resultados_df)
    estadistica_intervalos = calcular_estadistica_por_intervalo(
        resultados_df,
        stats_interval_seconds,
    )
    tabla_por_imagen = crear_tabla_por_imagen(resultados_df)

    estadistica_general.to_csv(stats_dir / "estadistica_general.csv", index=False)
    resumen_df.to_csv(stats_dir / "estadistica_por_persona.csv", index=False)
    estadistica_intervalos.to_csv(stats_dir / "estadistica_por_intervalos.csv", index=False)
    tabla_por_imagen.to_csv(stats_dir / "tabla_unica_por_imagen.csv", index=False)
    resultados_df.to_csv(stats_dir / "imagenes_analizadas_detalle.csv", index=False)

    print(f"Tablas estadisticas guardadas en: {stats_dir}")


def lineas_resumen_persona(resumen: pd.Series) -> list[str]:
    """Devuelve las lineas de texto para reportar una persona."""
    return [
        f"Persona ID: {int(resumen['persona_id'])}",
        f"Rostro representativo: {resumen['rostro_representativo']}",
        f"Detecciones: {int(resumen['detecciones'])}",
        f"Intervalo analizado: {resumen['inicio_segundo']}s a {resumen['fin_segundo']}s",
        f"Resultado dominante: {int(resumen['estres_dominante'])}",
        f"Confianza media: {resumen['confianza_media']}%",
        f"Porcentaje con estres: {resumen['porcentaje_estres']}%",
        f"Porcentaje sin estres: {resumen['porcentaje_sin_estres']}%",
    ]


def guardar_reportes_por_persona(
    resultados_df: pd.DataFrame,
    resumen_df: pd.DataFrame,
    output_csv: Path,
) -> None:
    """Guarda reportes detalle/resumen por persona."""
    reportes_dir = output_csv.parent / f"{output_csv.stem}_reportes_personas"
    asegurar_directorio(reportes_dir)

    for _, resumen in resumen_df.iterrows():
        persona_id = int(resumen["persona_id"])
        detalle = (
            resultados_df[resultados_df["persona_id"] == persona_id]
            .sort_values(by=["frame", "segundo"])
            .reset_index(drop=True)
        )
        persona_dir = reportes_dir / f"persona_{persona_id:03d}"
        asegurar_directorio(persona_dir)

        detalle.to_csv(persona_dir / "detalle.csv", index=False)
        pd.DataFrame([resumen.to_dict()]).to_csv(persona_dir / "resumen.csv", index=False)

        rostro_representativo = Path(str(resumen["rostro_representativo"]))
        if rostro_representativo.exists():
            shutil.copy2(rostro_representativo, persona_dir / "rostro_representativo.png")

        reporte_texto = "\n".join(lineas_resumen_persona(resumen))
        (persona_dir / "reporte.txt").write_text(reporte_texto, encoding="utf-8")

    print(f"Reportes individuales guardados en: {reportes_dir}")


def guardar_reporte_general(resumen_df: pd.DataFrame, output_csv: Path) -> None:
    """Genera un reporte general en texto con todas las personas detectadas."""
    reporte_path = output_csv.parent / f"{output_csv.stem}_reporte_general.txt"
    lineas = [
        "REPORTE GENERAL DE ESTRES POR PERSONA",
        "",
        f"Total de personas detectadas: {len(resumen_df)}",
        "",
    ]
    for _, resumen in resumen_df.iterrows():
        lineas.extend([*lineas_resumen_persona(resumen), ""])

    reporte_path.write_text("\n".join(lineas), encoding="utf-8")
    print(f"Reporte general guardado en: {reporte_path}")


def cargar_modelo(model_path: Path) -> tf.keras.Model:
    """Carga el modelo entrenado."""
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo indicado: {model_path}")
    print(f"Cargando modelo entrenado desde: {model_path}")
    model = tf.keras.models.load_model(model_path)
    validar_modelo_estres(model)
    return model


def validar_modelo_estres(model: tf.keras.Model) -> None:
    """Valida que el modelo sea compatible con el analisis de estres."""
    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)

    if input_shape[-3:] != (IMG_SIZE, IMG_SIZE, CHANNELS):
        raise ValueError(
            "El modelo cargado no tiene la entrada esperada. "
            f"Esperado: ({IMG_SIZE}, {IMG_SIZE}, {CHANNELS}); recibido: {input_shape}"
        )

    output_size = output_shape[-1]
    if output_size not in (len(STRESS_LABELS), len(STRESS_BY_MODEL_OUTPUT)):
        raise ValueError(
            "El modelo cargado no tiene una salida compatible. "
            f"Esperado: {len(STRESS_LABELS)} o {len(STRESS_BY_MODEL_OUTPUT)} clases; "
            f"recibido: {output_shape}"
        )


def imprimir_resumen_global(resultados_df: pd.DataFrame, resumen_df: pd.DataFrame) -> None:
    """Imprime un resumen del video analizado."""
    print("\n=== Resumen global del analisis de estres ===")
    print(resultados_df["estres_predicho"].value_counts())
    print("\n=== Resumen por rostro/persona ===")
    print(resumen_df.to_string(index=False))


def imprimir_configuracion_video(
    video_path: Path,
    total_frames: int,
    fps: float,
    frame_interval: int,
    min_confidence: float,
    strict_face_check: bool,
    faces_dir: Path,
) -> None:
    """Muestra la configuracion principal del analisis."""
    mensajes = [
        f"Analizando video: {video_path}",
        f"Frames totales: {total_frames}",
        f"FPS reportado: {fps:.2f}",
        f"Se analizara 1 de cada {frame_interval} frames",
        f"Confianza minima requerida: {min_confidence:.2f}%",
        f"Verificacion facial estricta: {'SI' if strict_face_check else 'NO'}",
        f"Los rostros extraidos se guardaran en: {faces_dir}",
    ]
    print("\n".join(mensajes))


def guardar_salidas(
    resultados_df: pd.DataFrame,
    resumen_df: pd.DataFrame,
    output_csv: Path,
    stats_interval_seconds: int,
) -> None:
    """Guarda todos los archivos de salida del analisis."""
    guardar_csv(resultados_df, output_csv)
    guardar_csv(resumen_df, output_csv.parent / f"{output_csv.stem}_resumen_personas.csv")
    guardar_reportes_por_persona(resultados_df, resumen_df, output_csv)
    guardar_reporte_general(resumen_df, output_csv)
    guardar_tablas_estadisticas(resultados_df, resumen_df, output_csv, stats_interval_seconds)


def analizar_video(
    model: tf.keras.Model,
    video_path: Path,
    output_csv: Path | None = None,
    frame_interval: int = 15,
    faces_dir: Path | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    strict_face_check: bool = True,
    stats_interval_seconds: int = STATS_INTERVAL_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analiza un video y calcula estres por persona."""
    if frame_interval <= 0:
        raise ValueError("--frame-interval debe ser mayor que 0.")
    if stats_interval_seconds <= 0:
        raise ValueError("--stats-interval debe ser mayor que 0.")
    if not video_path.exists():
        raise FileNotFoundError(f"No existe el video: {video_path}")

    cv2, detector, eye_detectors = cargar_detector_rostros()
    captura = cv2.VideoCapture(str(video_path))
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    if output_csv is None:
        output_csv = video_path.parent / f"{video_path.stem}_analisis.csv"
    if faces_dir is None:
        faces_dir = output_csv.parent / f"{output_csv.stem}_rostros"

    asegurar_directorio(faces_dir)

    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = captura.get(cv2.CAP_PROP_FPS) or 0.0
    trackers: dict[int, dict[str, float]] = {}
    siguiente_id = 1
    frame_idx = 0
    frames_analizados = 0
    resultados: list[dict[str, object]] = []
    descartados = 0

    imprimir_configuracion_video(
        video_path,
        total_frames,
        fps,
        frame_interval,
        min_confidence,
        strict_face_check,
        faces_dir,
    )

    while True:
        ok, frame = captura.read()
        if not ok:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detectar_rostros(gray, detector)
        segundo = frame_idx / fps if fps > 0 else 0.0
        ids_usados_frame: set[int] = set()

        for x, y, w, h in faces:
            rostro = gray[y : y + h, x : x + w]
            if not es_rostro_valido(
                rostro,
                (x, y, w, h),
                gray.shape,
                eye_detectors,
                strict_face_check,
            ):
                descartados += 1
                continue

            estres, confianza = predecir_estres(model, rostro, cv2)
            if confianza < min_confidence:
                descartados += 1
                continue

            persona_id, siguiente_id = asignar_persona_id(
                face_box=(x, y, w, h),
                trackers=trackers,
                siguiente_id=siguiente_id,
                frame_idx=frame_idx,
                max_distancia=80.0,
                max_salto_frames=frame_interval * 3,
                ids_usados_frame=ids_usados_frame,
            )

            rostro_path = guardar_rostro_video(
                cv2, rostro, faces_dir, persona_id, frame_idx, segundo
            )
            resultados.append(
                {
                    "video": video_path.name,
                    "frame": frame_idx,
                    "segundo": round(segundo, 2),
                    "persona_id": persona_id,
                    "rostro_path": rostro_path,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "estres_predicho": estres,
                    "confianza": round(confianza, 2),
                }
            )

        frames_analizados += 1
        if frames_analizados % 25 == 0 or (total_frames and frame_idx >= total_frames - 1):
            porcentaje = ((frame_idx + 1) / total_frames) * 100 if total_frames else 0
            print(
                f"Progreso video: frame {frame_idx + 1}/{total_frames} "
                f"({porcentaje:.2f}%) | rostros acumulados: {len(resultados)} "
                f"| descartados: {descartados}"
            )

        frame_idx += 1

    captura.release()

    resultados_df = pd.DataFrame(resultados)
    if resultados_df.empty:
        raise ValueError(
            "No se detectaron rostros validos en el video. "
            "Prueba con un video mas cercano, bajando el frame-interval, "
            "bajando --min-confidence o sin verificacion estricta."
        )

    resumen_df = resumir_video_por_persona(resultados_df)
    imprimir_resumen_global(resultados_df, resumen_df)
    guardar_salidas(resultados_df, resumen_df, output_csv, stats_interval_seconds)

    return resultados_df, resumen_df


def analizar_lote_videos(
    model: tf.keras.Model,
    video_paths: list[Path],
    output_dir: Path,
    frame_interval: int,
    min_confidence: float,
    strict_face_check: bool,
    stats_interval_seconds: int,
) -> None:
    """Analiza todos los videos encontrados y guarda salidas consolidadas."""
    videos = obtener_videos(video_paths)
    if not videos:
        raise FileNotFoundError("No se encontraron videos en las rutas indicadas.")

    asegurar_directorio(output_dir)
    resultados_lote = []
    resumenes_lote = []
    errores = []

    print(f"Videos encontrados: {len(videos)}")
    for video in videos:
        print(f"\n=== Analizando {video.name} ===")
        try:
            output_csv = output_dir / f"{video.stem}_analisis.csv"
            faces_dir = output_dir / f"{video.stem}_rostros"
            resultados_df, resumen_df = analizar_video(
                model,
                video,
                output_csv=output_csv,
                frame_interval=frame_interval,
                faces_dir=faces_dir,
                min_confidence=min_confidence,
                strict_face_check=strict_face_check,
                stats_interval_seconds=stats_interval_seconds,
            )
            resultados_lote.append(resultados_df)
            resumenes_lote.append(resumen_df.assign(video=video.name))
        except Exception as exc:
            errores.append({"video": video.name, "error": str(exc)})
            print(f"No se pudo analizar {video.name}: {exc}")

    if resultados_lote:
        guardar_csv(pd.concat(resultados_lote, ignore_index=True), output_dir / "lote_detalle.csv")
        guardar_csv(pd.concat(resumenes_lote, ignore_index=True), output_dir / "lote_resumen_personas.csv")
    if errores:
        guardar_csv(pd.DataFrame(errores), output_dir / "lote_errores.csv")


def parse_args() -> argparse.Namespace:
    """Define los argumentos del script."""
    parser = argparse.ArgumentParser(
        description="Analiza un video y estima si hay presencia de estres por persona."
    )
    parser.add_argument("--load-model", type=Path, required=True, help="Ruta al modelo.")
    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        nargs="+",
        help="Ruta a uno o varios videos, o carpetas con videos.",
    )
    parser.add_argument("--output-dir", type=Path, help="Carpeta para analisis de varios videos.")
    parser.add_argument("--output-csv", type=Path, help="Ruta para guardar el detalle.")
    parser.add_argument("--frame-interval", type=int, default=15, help="Analiza 1 de cada N frames.")
    parser.add_argument("--faces-dir", type=Path, help="Carpeta para rostros extraidos.")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE, help="Confianza minima.")
    parser.add_argument("--stats-interval", type=int, default=STATS_INTERVAL_SECONDS, help="Segundos por intervalo estadistico.")
    face_check_group = parser.add_mutually_exclusive_group()
    face_check_group.add_argument("--strict-face-check", action="store_true", default=True)
    face_check_group.add_argument("--no-strict-face-check", action="store_false", dest="strict_face_check")
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal."""
    args = parse_args()
    model = cargar_modelo(args.load_model)

    if len(args.video_path) > 1 or args.video_path[0].is_dir():
        analizar_lote_videos(
            model,
            args.video_path,
            output_dir=args.output_dir or Path("analisis_lote_videos"),
            frame_interval=args.frame_interval,
            min_confidence=args.min_confidence,
            strict_face_check=args.strict_face_check,
            stats_interval_seconds=args.stats_interval,
        )
        return

    analizar_video(
        model,
        args.video_path[0],
        output_csv=args.output_csv,
        frame_interval=args.frame_interval,
        faces_dir=args.faces_dir,
        min_confidence=args.min_confidence,
        strict_face_check=args.strict_face_check,
        stats_interval_seconds=args.stats_interval,
    )


if __name__ == "__main__":
    main()
