# previsor.py

import pandas as pd
import statsmodels.api as sm
import numpy as np
import pickle
import os
import shutil
from api import set_training_status

def treinar_modelos_de_arquivo(caminho_do_arquivo):
    """
    Função de treino pesada. Será executada em segundo plano (thread).
    """
    pasta_modelos = 'modelos_salvos'
    print(f"THREAD: A iniciar treino a partir do ficheiro {caminho_do_arquivo}...")

    if os.path.exists(pasta_modelos):
        shutil.rmtree(pasta_modelos)
    os.makedirs(pasta_modelos)

    try:
        lojas_unicas = pd.read_csv(caminho_do_arquivo, delimiter=';', usecols=['LOJNUMERO'])['LOJNUMERO'].unique()
        print(f"THREAD: Encontradas {len(lojas_unicas)} lojas.")

        for loja_id in lojas_unicas:
            try:
                df_loja = pd.read_csv(caminho_do_arquivo, delimiter=';')
                df_loja = df_loja[df_loja['LOJNUMERO'] == loja_id].copy()

                if len(df_loja) < 12: continue

                df_loja['DATA'] = pd.to_datetime(df_loja['ANO'].astype(str) + '-' + df_loja['MES'].astype(str) + '-01')
                df_loja = df_loja.set_index('DATA').sort_index()
                df_loja['SOMA_LOG'] = np.log1p(df_loja['SOMA'])
                
                modelo_soma = sm.tsa.SARIMAX(df_loja['SOMA_LOG'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
                
                with open(f'{pasta_modelos}/modelo_loja_{loja_id}_soma.pkl', 'wb') as pkl:
                    pickle.dump(modelo_soma, pkl)
            except Exception as e:
                print(f"THREAD: ERRO ao treinar a Loja {loja_id}: {e}")
            finally:
                if 'df_loja' in locals(): del df_loja
        
        with open(os.path.join(pasta_modelos, '_SUCESSO.txt'), 'w') as f:
            f.write('Treino concluído.')
        print("THREAD: Treino concluído com sucesso.")
    except Exception as e:
        print(f"THREAD: ERRO CRÍTICO no processo de treino: {e}")
    finally:
        # Informa a API principal que o treino terminou
        set_training_status(False)


def obter_previsao(loja_desejada, mes_desejado, ano_desejado):
    pasta_modelos = 'modelos_salvos'
    caminho_modelo_soma = f'{pasta_modelos}/modelo_loja_{loja_desejada}_soma.pkl'

    if not os.path.exists(caminho_modelo_soma):
        return {"erro": "Modelos para esta loja não foram encontrados. O treino pode estar em andamento ou ter falhado."}
    
    # ... (O resto da função de previsão permanece igual)
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
        return {"erro": f"Ocorreu um erro ao gerar a previsão: {e}"}# previsor.py

import pandas as pd
import statsmodels.api as sm
import numpy as np
import pickle
import os
import shutil
from api import set_training_status

def treinar_modelos_de_arquivo(caminho_do_arquivo):
    """
    Função de treino pesada. Será executada em segundo plano (thread).
    """
    pasta_modelos = 'modelos_salvos'
    print(f"THREAD: A iniciar treino a partir do ficheiro {caminho_do_arquivo}...")

    if os.path.exists(pasta_modelos):
        shutil.rmtree(pasta_modelos)
    os.makedirs(pasta_modelos)

    try:
        lojas_unicas = pd.read_csv(caminho_do_arquivo, delimiter=';', usecols=['LOJNUMERO'])['LOJNUMERO'].unique()
        print(f"THREAD: Encontradas {len(lojas_unicas)} lojas.")

        for loja_id in lojas_unicas:
            try:
                df_loja = pd.read_csv(caminho_do_arquivo, delimiter=';')
                df_loja = df_loja[df_loja['LOJNUMERO'] == loja_id].copy()

                if len(df_loja) < 12: continue

                df_loja['DATA'] = pd.to_datetime(df_loja['ANO'].astype(str) + '-' + df_loja['MES'].astype(str) + '-01')
                df_loja = df_loja.set_index('DATA').sort_index()
                df_loja['SOMA_LOG'] = np.log1p(df_loja['SOMA'])
                
                modelo_soma = sm.tsa.SARIMAX(df_loja['SOMA_LOG'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
                
                with open(f'{pasta_modelos}/modelo_loja_{loja_id}_soma.pkl', 'wb') as pkl:
                    pickle.dump(modelo_soma, pkl)
            except Exception as e:
                print(f"THREAD: ERRO ao treinar a Loja {loja_id}: {e}")
            finally:
                if 'df_loja' in locals(): del df_loja
        
        with open(os.path.join(pasta_modelos, '_SUCESSO.txt'), 'w') as f:
            f.write('Treino concluído.')
        print("THREAD: Treino concluído com sucesso.")
    except Exception as e:
        print(f"THREAD: ERRO CRÍTICO no processo de treino: {e}")
    finally:
        # Informa a API principal que o treino terminou
        set_training_status(False)


def obter_previsao(loja_desejada, mes_desejado, ano_desejado):
    pasta_modelos = 'modelos_salvos'
    caminho_modelo_soma = f'{pasta_modelos}/modelo_loja_{loja_desejada}_soma.pkl'

    if not os.path.exists(caminho_modelo_soma):
        return {"erro": "Modelos para esta loja não foram encontrados. O treino pode estar em andamento ou ter falhado."}
    
    # ... (O resto da função de previsão permanece igual)
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
