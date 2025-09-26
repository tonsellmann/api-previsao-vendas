# tasks.py (ou previsor.py, se for o caso)

import pandas as pd
import statsmodels.api as sm
import numpy as np
import pickle
import os
import shutil

# ... (o resto do código é o mesmo até à função de treino) ...

def treinar_modelos_de_arquivo(caminho_do_arquivo):
    """
    Função de treino pesada. Corrigida para tratar o ID da loja como inteiro.
    """
    pasta_modelos = 'modelos_salvos'
    print(f"WORKER: A iniciar treino a partir do ficheiro {caminho_do_arquivo}...")

    if os.path.exists(pasta_modelos):
        shutil.rmtree(pasta_modelos)
    os.makedirs(pasta_modelos)

    try:
        # --- CORREÇÃO APLICADA AQUI ---
        # Forçamos a coluna LOJNUMERO a ser lida como um número inteiro.
        lojas_unicas = pd.read_csv(caminho_do_arquivo, delimiter=';', usecols=['LOJNUMERO'], dtype={'LOJNUMERO': int})['LOJNUMERO'].unique()
        print(f"WORKER: Encontradas {len(lojas_unicas)} lojas. A iniciar treino sequencial...")

        lojas_treinadas = 0
        for loja_id in lojas_unicas:
            try:
                # --- CORREÇÃO APLICADA AQUI TAMBÉM ---
                df_loja = pd.read_csv(caminho_do_arquivo, delimiter=';', dtype={'LOJNUMERO': int})
                df_loja = df_loja[df_loja['LOJNUMERO'] == loja_id].copy()

                if len(df_loja) < 12:
                    continue

                df_loja['DATA'] = pd.to_datetime(df_loja['ANO'].astype(str) + '-' + df_loja['MES'].astype(str) + '-01')
                df_loja = df_loja.set_index('DATA').sort_index()
                df_loja['SOMA_LOG'] = np.log1p(df_loja['SOMA'])
                
                modelo_soma = sm.tsa.SARIMAX(df_loja['SOMA_LOG'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
                
                with open(f'{pasta_modelos}/modelo_loja_{loja_id}_soma.pkl', 'wb') as pkl:
                    pickle.dump(modelo_soma, pkl)
                
                lojas_treinadas += 1
            except Exception as e:
                print(f"WORKER: ERRO ao treinar a Loja {loja_id}: {e}")
            finally:
                if 'df_loja' in locals(): del df_loja
        
        with open(os.path.join(pasta_modelos, '_SUCESSO.txt'), 'w') as f:
            f.write(f'Treino concluído com sucesso para {lojas_treinadas} lojas.')
        print(f"WORKER: Treino concluído. {lojas_treinadas} lojas processadas.")
        return True

    except Exception as e:
        print(f"WORKER: ERRO CRÍTICO no processo de treino: {e}")
        return False

# Adicione esta função ao seu previsor.py se estiver a usar a versão com threading
def obter_previsao(loja_desejada, mes_desejado, ano_desejado):
    # A função de previsão não precisa de alterações
    # ...
