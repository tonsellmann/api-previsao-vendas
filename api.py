# api.py

from flask import Flask, request, jsonify
import threading
import os
from previsor import treinar_modelos_de_arquivo, obter_previsao

app = Flask(__name__)
PASTA_UPLOADS = 'uploads'
os.makedirs(PASTA_UPLOADS, exist_ok=True)

# Variável global para controlar o status do treino
TREINO_EM_ANDAMENTO = False

def set_training_status(status):
    """Função para alterar o status do treino de forma segura."""
    global TREINO_EM_ANDAMENTO
    TREINO_EM_ANDAMENTO = status

@app.route('/treinar', methods=['POST'])
def treinar():
    global TREINO_EM_ANDAMENTO
    if TREINO_EM_ANDAMENTO:
        return jsonify({"erro": "Um processo de treino já está em andamento."}), 409

    if 'file' not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado."}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"erro": "Formato de arquivo inválido."}), 400

    caminho_arquivo = os.path.join(PASTA_UPLOADS, file.filename)
    file.save(caminho_arquivo)

    # Inicia o treino em segundo plano e atualiza o status
    set_training_status(True)
    thread_de_treino = threading.Thread(target=treinar_modelos_de_arquivo, args=(caminho_arquivo,))
    thread_de_treino.start()

    return jsonify({"mensagem": "Arquivo recebido. O treino foi iniciado em segundo plano."}), 202

@app.route('/prever', methods=['GET'])
def prever():
    if TREINO_EM_ANDAMENTO:
        return jsonify({"erro": "A API está ocupada a treinar novos modelos. Por favor, tente novamente em alguns minutos."}), 503

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
    pasta_modelos = 'modelos_salvos'
    status_final = "inativo_ou_concluido"
    
    if TREINO_EM_ANDAMENTO:
        status_final = "em_andamento"
    
    modelos_gerados = []
    if os.path.exists(pasta_modelos):
        modelos_gerados = os.listdir(pasta_modelos)

    return jsonify({
        "status_treino_atual": status_final,
        "modelos_na_memoria": modelos_gerados
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
