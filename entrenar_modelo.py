# -*- coding: utf-8 -*-
"""Entrenamiento del modelo de análisis de estrés usando FER2013.

Este script entrena una CNN para detectar emociones en rostros y estimar
el nivel de estrés asociado.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


IMG_SIZE = 48
CHANNELS = 1
NUM_CLASSES = 7
NORM_FACTOR = 255.0

EMOCIONES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

NIVEL_ESTRES = {
    0: "ALTO",
    1: "ALTO",
    2: "ALTO",
    3: "BAJO",
    4: "MEDIO",
    5: "BAJO",
    6: "BAJO"
}


def cargar_datos_carpeta(train_dir: Path, img_size: int = IMG_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Carga datos desde carpetas de imágenes."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        color_mode="grayscale",
        class_mode="sparse",
        batch_size=32,
        shuffle=True
    )

    X, y = [], []
    for i in range(len(train_data)):
        batch_x, batch_y = train_data[i]
        X.extend(batch_x)
        y.extend(batch_y)

    X = np.array(X)
    y = np.array(y)

    print(f"Imágenes cargadas: {X.shape[0]}")
    print(f"Forma etiquetas: {y.shape}")
    print(f"Clases detectadas: {train_data.class_indices}")

    return X, y


def crear_modelo() -> models.Model:
    """Crea el modelo CNN simple y estable."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def entrenar(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    callbacks_list: list | None = None,
) -> models.Model:
    """Entrena el modelo sin augmentation."""
    model = crear_modelo()

    print("\n=== Arquitectura del modelo ===")
    model.summary()

    print(f"\n=== Entrenamiento ===")
    print(f"Entradas entrenamiento: {X_train.shape[0]}")
    print(f"Entradas validación: {X_val.shape[0]}")
    print(f"Épocas: {epochs}, Batch size: {batch_size}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )

    return model


def mostrar_resultados(history: tf.keras.callbacks.History, model: models.Model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Muestra métricas y distribución de predicciones."""
    print("\n=== Resultados finales ===")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Pérdida test: {loss:.4f}")
    print(f"Precisión test: {accuracy:.4f}")

    predicciones = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predicciones, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    print("\n=== Distribución de predicciones ===")
    for i, nombre in EMOCIONES.items():
        count = np.sum(pred_classes == i)
        pct = count / len(pred_classes) * 100
        print(f"  {nombre}: {count} ({pct:.1f}%)")

    print("\n=== Distribución de estrés (mapeado) ===")
    stres_map = {0: "ALTO", 1: "ALTO", 2: "ALTO", 3: "BAJO", 4: "MEDIO", 5: "BAJO", 6: "BAJO"}
    for nivel in ["BAJO", "MEDIO", "ALTO"]:
        count = sum(np.sum(pred_classes == e) for e, v in stres_map.items() if v == nivel)
        pct = count / len(pred_classes) * 100
        print(f"  {nivel}: {count} ({pct:.1f}%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Entrena modelo de análisis de estrés")
    parser.add_argument("--input", type=Path, help="Ruta a carpeta de imágenes o CSV")
    parser.add_argument("--csv", action="store_true", help="Indica que la entrada es un CSV")
    parser.add_argument("--output", type=Path, default=Path("modelo.keras"), help="Ruta para guardar el modelo")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño del batch")
    args = parser.parse_args()

    print("=== Cargando datos ===")

    if args.csv:
        X, y = cargar_datos_csv(args.input)
    else:
        X, y = cargar_datos_carpeta(args.input)

    y = to_categorical(y, num_classes=NUM_CLASSES)

    print(f"Forma X: {X.shape}")
    print(f"Forma y: {y.shape}")

    y_labels = np.argmax(y, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=np.argmax(y_train, axis=1)
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    model = entrenar(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks_list=[early_stop, reduce_lr]
    )

    mostrar_resultados(model.history, model, X_test, y_test)

    model.save(args.output)
    print(f"\nModelo guardado en: {args.output}")


if __name__ == "__main__":
    main()