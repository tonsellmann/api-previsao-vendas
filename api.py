# api.py (Versão Final e Simples)

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

@app.route('/prever', methods=['GET'])
def prever():
    pasta_modelos = 'modelos_salvos'
    try:
        loja = int(request.args.get('loja'))
        mes = int(request.args.get('mes'))
        ano = int(request.args.get('ano'))
    except (TypeError, ValueError):
        return jsonify({"erro": "Parâmetros 'loja', 'mes' e 'ano' são obrigatórios e devem ser números inteiros."}), 400

    caminho_modelo = f'{pasta_modelos}/modelo_loja_{loja}_soma.pkl'

    if not os.path.exists(caminho_modelo):
        return jsonify({"erro": f"Modelo para a Loja {loja} não encontrado. Verifique o status do treino no endpoint /status."}), 404

    try:
        with open(caminho_modelo, 'rb') as pkl:
            modelo = pickle.load(pkl)

        data_previsao = pd.to_datetime(f'{ano}-{mes}-01')
        predicao_log = modelo.get_prediction(start=data_previsao, end=data_previsao)
        valor_previsto = round(np.expm1(predicao_log.predicted_mean.iloc[0]), 2)
        
        return jsonify({
            "loja_consultada": loja,
            "previsao_para_data": f"{mes:02d}/{ano}",
            "previsao_valor_vendas": float(max(0, valor_previsto)),
            "status": "sucesso"
        })
    except Exception as e:
        return jsonify({"erro": f"Ocorreu um erro ao gerar a previsão: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def status():
    """Endpoint de diagnóstico para verificar o resultado do treino."""
    pasta_modelos = 'modelos_salvos'
    status_treino = "FALHOU"
    detalhes = "A pasta 'modelos_salvos' não foi encontrada."
    
    if os.path.exists(os.path.join(pasta_modelos, '_SUCESSO.txt')):
        status_treino = "SUCESSO"
        try:
            with open(os.path.join(pasta_modelos, '_SUCESSO.txt'), 'r') as f:
                detalhes = f.read()
        except Exception:
            detalhes = "Ficheiro de sucesso encontrado, mas não pôde ser lido."

    return jsonify({
        "status_do_treino_na_publicacao": status_treino,
        "detalhes": detalhes
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
