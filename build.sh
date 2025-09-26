#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "--- INÍCIO DO BUILD SCRIPT ---"

echo "PASSO 1: A instalar as dependências..."
pip install -r requirements.txt

echo "PASSO 2: A executar o script de treino de modelos..."
python treinar_modelos.py

echo "--- BUILD SCRIPT CONCLUÍDO COM SUCESSO ---"