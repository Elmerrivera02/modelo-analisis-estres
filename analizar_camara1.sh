#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python proyecto_tesis.py \
  --load-model modelo.keras \
  --video-path camara1.avi camara1.1.avi \
  --output-dir analisis_camara1_comparacion \
  --frame-interval 15 \
  --min-confidence 25 \
  --stats-interval 10
