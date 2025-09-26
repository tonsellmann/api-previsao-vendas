# treinar_modelos.py

import pandas as pd
import statsmodels.api as sm
import numpy as np
import pickle
import os
import shutil

def treinar_e_salvar_modelos():
    print("--- INÍCIO DO SCRIPT DE TREINO ---")
    arquivo_csv = 'vendas.csv'
    pasta_modelos = 'modelos_salvos'

    if os.path.exists(pasta_modelos):
        shutil.rmtree(pasta_modelos)
    os.makedirs(pasta_modelos)

    print(f"A carregar dados de '{arquivo_csv}'...")
    df = pd.read_csv(arquivo_csv, delimiter=';')
    df['DATA'] = pd.to_datetime(df['ANO'].astype(str) + '-' + df['MES'].astype(str) + '-01')
    df = df.set_index('DATA')

    lojas_unicas = df['LOJNUMERO'].unique()
    print(f"Encontradas {len(lojas_unicas)} lojas. A iniciar treino...")

    for loja_id in lojas_unicas:
        try:
            print(f"  A treinar modelos para a Loja {loja_id}...")
            df_loja = df[df['LOJNUMERO'] == loja_id].copy().sort_index()
            if len(df_loja) < 12:
                print(f"    AVISO: Loja {loja_id} ignorada (dados insuficientes).")
                continue

            df_loja['SOMA_LOG'] = np.log1p(df_loja['SOMA'])
            modelo_soma = sm.tsa.SARIMAX(df_loja['SOMA_LOG'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
            with open(f'{pasta_modelos}/modelo_loja_{loja_id}_soma.pkl', 'wb') as pkl:
                pickle.dump(modelo_soma, pkl)
            
            print(f"    Modelo para a Loja {loja_id} treinado e salvo.")
        except Exception as e:
            print(f"    !!! ERRO CRÍTICO ao treinar para a Loja {loja_id}: {e}")

    # Cria um arquivo de sucesso para a API poder verificar
    with open(os.path.join(pasta_modelos, '_SUCESSO.txt'), 'w') as f:
        f.write('Treino concluído com sucesso.')
    
    print("--- FIM DO SCRIPT DE TREINO ---")

if __name__ == '__main__':
    treinar_e_salvar_modelos()
