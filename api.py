# api.py (Versão Corrigida - Sem Importação Circular)

from flask import Flask, request, jsonify
import threading
import os
from previsor import treinar_modelos_de_arquivo, obter_previsao

app = Flask(__name__)
PASTA_UPLOADS = 'uploads'
os.makedirs(PASTA_UPLOADS, exist_ok=True)

# A variável de estado vive apenas aqui, no ficheiro principal da API.
TREINO_EM_ANDAMENTO = False

def wrapper_de_treino(caminho_arquivo):
    """
    Função "embrulho" que executa o treino e atualiza o estado da API
    quando este termina.
    """
    global TREINO_EM_ANDAMENTO
    try:
        # Chama a função de treino do outro ficheiro
        treinar_modelos_de_arquivo(caminho_arquivo)
    finally:
        # Aconteça o que acontecer (sucesso ou erro), diz à API que o treino terminou.
        TREINO_EM_ANDAMENTO = False


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
    TREINO_EM_ANDAMENTO = True
    # A thread agora chama a nossa função "embrulho"
    thread_de_treino = threading.Thread(target=wrapper_de_treino, args=(caminho_arquivo,))
    thread_de_treino.start()

    return jsonify({"mensagem": "Arquivo recebido. O treino foi iniciado em segundo plano."}), 202

@app.route('/prever', methods=['GET'])
def prever():
    if TREINO_EM_ANDAMENTO:
        return jsonify({"erro": "A API está ocupada a treinar. Tente novamente em alguns minutos."}), 503

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
    status_final = "inativo_ou_concluido"
    if TREINO_EM_ANDAMENTO:
        status_final = "em_andamento"
    return jsonify({"status_treino_atual": status_final})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
