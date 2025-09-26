# previsor.py

import pandas as pd
import numpy as np
import pickle
import os

def obter_previsao(loja_desejada, mes_desejado, ano_desejado):
    pasta_modelos = 'modelos_salvos'
    caminho_modelo_soma = f'{pasta_modelos}/modelo_loja_{loja_desejada}_soma.pkl'

    if not os.path.exists(caminho_modelo_soma):
        return {"erro": f"Modelos para a Loja {loja_desejada} não encontrados. Verifique o status do treino."}

    try:
        with open(caminho_modelo_soma, 'rb') as pkl:
            modelo_soma = pickle.load(pkl)

        data_previsao = pd.to_datetime(f'{ano_desejado}-{mes_desejado}-01')
        pred_soma_log = modelo_soma.get_prediction(start=data_previsao, end=data_previsao)
        valor_previsto_soma = round(np.expm1(pred_soma_log.predicted_mean.iloc[0]), 2)
        
        return {
            "loja_consultada": loja_desejada,
            "previsao_para_data": f"{mes_desejado:02d}/{ano_desejado}",
            "previsao_valor_vendas": float(max(0, valor_previsto_soma)),
            "status": "sucesso"
        }
    except Exception as e:
        return {"erro": f"Ocorreu um erro ao gerar a previsão: {e}"}
