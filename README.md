# Analisis de estres en videos

Proyecto de tesis para analizar rostros en videos y estimar presencia de estres de forma binaria:

- `1`: con estres
- `0`: sin estres

El script principal es `proyecto_tesis.py`.

## Requisitos

Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verificar sintaxis

```bash
.venv/bin/python -m py_compile proyecto_tesis.py
```

Si no aparece ningun mensaje, el archivo esta correcto.

## Ejecutar analisis

Analizar `camara1.avi` y `camara1.1.avi` juntos:

```bash
./analizar_camara1.sh
```

Analizar el conjunto configurado del proyecto:

```bash
./analizar_todos_videos.sh
```

## Salidas generadas

El analisis genera archivos CSV, reportes de texto, rostros extraidos y tablas estadisticas.

La tabla mas directa por imagen se encuentra en:

```text
*_tablas_estadisticas/tabla_unica_por_imagen.csv
```

Columnas principales:

- `video`
- `frame`
- `segundo`
- `persona_id`
- `rostro_path`
- `estres_predicho`
- `confianza`
- `descripcion`

## Archivos no incluidos en GitHub

Los videos, datasets grandes, carpetas de resultados y el entorno virtual estan excluidos mediante `.gitignore`.

Coloca localmente los videos y el archivo `modelo.keras` antes de ejecutar el analisis.
