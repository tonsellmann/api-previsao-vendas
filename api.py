# api.py (Versão Autónoma)

from flask import Flask, request, jsonify
# threading permite executar tarefas em segundo plano
import threading
import os
from previsor import treinar_modelos_de_arquivo, obter_previsao, TREINO_EM_ANDAMENTO

app = Flask(__name__)
PASTA_UPLOADS = 'uploads'
app.config['PASTA_UPLOADS'] = PASTA_UPLOADS

# Cria a pasta para os uploads, se não existir
if not os.path.exists(PASTA_UPLOADS):
    os.makedirs(PASTA_UPLOADS)

@app.route('/treinar', methods=['POST'])
def treinar():
    """
    Endpoint para receber um arquivo CSV e iniciar o treino em segundo plano.
    """
    # Usa a variável global importada do previsor
    if TREINO_EM_ANDAMENTO:
        return jsonify({"erro": "Um processo de treino já está em andamento. Por favor, aguarde a sua conclusão."}), 409 # 409 Conflict

    if 'file' not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado. Por favor, envie um arquivo CSV no campo 'file'."}), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({"erro": "Nome de arquivo inválido ou não é um CSV."}), 400

    # Salva o arquivo no servidor
    caminho_arquivo = os.path.join(app.config['PASTA_UPLOADS'], file.filename)
    file.save(caminho_arquivo)

    # Inicia o treino em uma thread separada (em segundo plano)
    thread_de_treino = threading.Thread(target=treinar_modelos_de_arquivo, args=(caminho_arquivo,))
    thread_de_treino.start()

    # Responde imediatamente
    return jsonify({
        "status": "sucesso",
        "mensagem": f"Arquivo '{file.filename}' recebido. O treino dos modelos foi iniciado em segundo plano. As previsões estarão disponíveis em alguns minutos."
    }), 202 # 202 Accepted

@app.route('/prever', methods=['GET'])
def prever():
    """
    Endpoint para obter uma previsão, usando os modelos já treinados.
    """
    if TREINO_EM_ANDAMENTO:
        return jsonify({"erro": "A API está ocupada a treinar novos modelos. Por favor, tente novamente em alguns minutos."}), 503 # 503 Service Unavailable

    try:
        loja = int(request.args.get('loja'))
        mes = int(request.args.get('mes'))
        ano = int(request.args.get('ano'))
    except (TypeError, ValueError):
        return jsonify({"erro": "Parâmetros 'loja', 'mes' e 'ano' são obrigatórios e devem ser números inteiros."}), 400

    resultado = obter_previsao(loja, mes, ano)

    if "erro" in resultado:
        return jsonify(resultado), 404 # 404 Not Found (se o modelo da loja não existir)
    
    return jsonify(resultado)

@app.route('/status', methods=['GET'])
def status():
    """
    Endpoint opcional para verificar se um treino ainda está a decorrer.
    """
    if TREINO_EM_ANDAMENTO:
        return jsonify({"status_treino": "em_andamento"})
    else:
        return jsonify({"status_treino": "concluido_ou_inativo"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
