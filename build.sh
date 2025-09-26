#!/usr/bin/env bash
set -o errexit
echo "--- INÍCIO DO BUILD SCRIPT ---"
pip install -r requirements.txt
echo "--- A EXECUTAR O TREINO ---"
python treinar_modelos.py
echo "--- BUILD SCRIPT CONCLUÍDO ---"