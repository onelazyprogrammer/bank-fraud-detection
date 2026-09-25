#!/usr/bin/env bash
# Levanta la app de Streamlit. Crea el entorno conda si no existe.
set -euo pipefail

ENTORNO="diplomado-mldl"

cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: no se encontró conda. Instalá Miniconda o Anaconda primero." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda run -n "$ENTORNO" python --version >/dev/null 2>&1; then
    echo "Creando el entorno conda '$ENTORNO'..."
    conda create -y -n "$ENTORNO" python=3.11
fi

conda activate "$ENTORNO"

echo "Instalando dependencias..."
pip install -q -r requirements.txt

if [ ! -f model.pkl ] || [ ! -f datos_muestra.csv ]; then
    echo "Error: faltan model.pkl o datos_muestra.csv. Ejecutá notebooks/notebook_final.ipynb primero." >&2
    exit 1
fi

streamlit run app.py
