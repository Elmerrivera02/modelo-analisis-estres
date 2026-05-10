#!/usr/bin/env bash
set -uo pipefail

ERRORES="analisis_ejecucion_errores.csv"
printf "analisis,codigo_salida\n" > "$ERRORES"

ejecutar() {
  local nombre="$1"
  shift

  echo
  echo "=== Ejecutando: $nombre ==="
  if "$@"; then
    echo "OK: $nombre"
  else
    local codigo=$?
    echo "ERROR: $nombre termino con codigo $codigo"
    printf "%s,%s\n" "$nombre" "$codigo" >> "$ERRORES"
  fi
}

ejecutar "camara1_comparacion" \
  .venv/bin/python proyecto_tesis.py \
    --load-model modelo.keras \
    --video-path camara1.avi camara1.1.avi \
    --output-dir analisis_camara1_comparacion \
    --frame-interval 15 \
    --min-confidence 25 \
    --stats-interval 10

ejecutar "camara2" \
  .venv/bin/python proyecto_tesis.py \
    --load-model modelo.keras \
    --video-path camara2.avi \
    --output-csv analisis_camara2/camara2_analisis.csv \
    --faces-dir analisis_camara2/camara2_rostros \
    --frame-interval 15 \
    --min-confidence 25 \
    --stats-interval 10

ejecutar "video_prueba" \
  .venv/bin/python proyecto_tesis.py \
    --load-model modelo.keras \
    --video-path Video_Prueba.mp4 \
    --output-csv analisis_video_prueba/video_prueba_analisis.csv \
    --faces-dir analisis_video_prueba/video_prueba_rostros \
    --frame-interval 15 \
    --min-confidence 25 \
    --stats-interval 10

echo
echo "Ejecucion completa. Si algun analisis fallo, revisa: $ERRORES"
