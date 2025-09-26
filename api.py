# api.py

from flask import Flask, request, jsonify
import os
from redis import Redis
from rq import Queue
from tasks import treinar_modelos_de_arquivo

app = Flask(__name__)
PASTA_UPLOADS = 'uploads'
os.makedirs(PASTA_UPLOADS, exist_ok=True)

# Conecta-se à fila de tarefas (o Render irá fornecer este URL)
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
conn = Redis.from_url(redis_url)
q = Queue(connection=conn)

@app.route('/treinar', methods=['POST'])
def treinar():
    if 'file' not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado."}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"erro": "Formato de arquivo inválido. Por favor, envie um .csv"}), 400

    caminho_arquivo = os.path.join(PASTA_UPLOADS, file.filename)
    file.save(caminho_arquivo)

    # Adiciona a tarefa de treino à fila para ser executada pelo worker
    q.enqueue(treinar_modelos_de_arquivo, caminho_arquivo)

    return jsonify({"mensagem": "Arquivo recebido. O treino foi iniciado em segundo plano."}), 202

# Os outros endpoints (prever, status) podem ser adicionados aqui
# Mas vamos focar em fazer o treino funcionar primeiro.

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
