# api.py

from flask import Flask, request, jsonify
from previsor import obter_previsao
import os

app = Flask(__name__)

@app.route('/prever', methods=['GET'])
def prever():
    try:
        loja = int(request.args.get('loja'))
        mes = int(request.args.get('mes'))
        ano = int(request.args.get('ano'))
    except (TypeError, ValueError):
        return jsonify({"erro": "Parâmetros inválidos."}), 400
    
    resultado = obter_previsao(loja, mes, ano)
    return jsonify(resultado)

@app.route('/status', methods=['GET'])
def status():
    """
    Endpoint de diagnóstico para verificar o resultado do treino.
    """
    pasta_modelos = 'modelos_salvos'
    status_treino = "FALHOU"
    arquivos_modelo = []

    if os.path.exists(os.path.join(pasta_modelos, '_SUCESSO.txt')):
        status_treino = "SUCESSO"
    
    if os.path.exists(pasta_modelos):
        arquivos_modelo = os.listdir(pasta_modelos)

    return jsonify({
        "status_do_treino_automatico": status_treino,
        "numero_de_modelos_gerados": len(arquivos_modelo),
        "modelos_gerados": arquivos_modelo
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
