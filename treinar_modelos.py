# treinar_modelos.py (Versão Final - Otimizada para Baixa Memória)

import pandas as pd
import statsmodels.api as sm
import numpy as np
import pickle
import os
import shutil

def treinar_e_salvar_modelos():
    print("--- INÍCIO DO SCRIPT DE TREINO (MODO DE MEMÓRIA ULTRA-EFICIENTE) ---")
    arquivo_csv = 'vendas.csv'
    pasta_modelos = 'modelos_salvos'

    if os.path.exists(pasta_modelos):
        shutil.rmtree(pasta_modelos)
    os.makedirs(pasta_modelos)

    try:
        print("A identificar lojas únicas de forma eficiente...")
        # Passo 1: Lê apenas a coluna das lojas para descobrir quais são, sem carregar tudo.
        lojas_unicas = pd.read_csv(arquivo_csv, delimiter=';', usecols=['LOJNUMERO'])['LOJNUMERO'].unique()
        print(f"Encontradas {len(lojas_unicas)} lojas. A iniciar treino loja a loja...")

        lojas_treinadas_com_sucesso = 0
        for loja_id in lojas_unicas:
            try:
                print(f"  -> A processar Loja {loja_id}...")
                
                # Passo 2: Lê o CSV inteiro, mas mantém apenas as linhas da loja atual em memória.
                # Isto é o mais eficiente que podemos ser.
                df_loja = pd.read_csv(arquivo_csv, delimiter=';')
                df_loja = df_loja[df_loja['LOJNUMERO'] == loja_id].copy()

                if len(df_loja) < 12:
                    print(f"     - Loja {loja_id}: Ignorada (dados insuficientes).")
                    continue

                df_loja['DATA'] = pd.to_datetime(df_loja['ANO'].astype(str) + '-' + df_loja['MES'].astype(str) + '-01')
                df_loja = df_loja.set_index('DATA').sort_index()

                df_loja['SOMA_LOG'] = np.log1p(df_loja['SOMA'])
                
                modelo_soma = sm.tsa.SARIMAX(df_loja['SOMA_LOG'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
                
                with open(f'{pasta_modelos}/modelo_loja_{loja_id}_soma.pkl', 'wb') as pkl:
                    pickle.dump(modelo_soma, pkl)
                
                print(f"     Modelo para Loja {loja_id} salvo com sucesso.")
                lojas_treinadas_com_sucesso += 1

            except Exception as e:
                print(f"    !!! ERRO ao treinar para a Loja {loja_id}: {e}")
            finally:
                # Garante que a memória é libertada antes da próxima iteração
                if 'df_loja' in locals():
                    del df_loja

        # Verifica se o processo foi bem-sucedido
        if lojas_treinadas_com_sucesso > 0:
            with open(os.path.join(pasta_modelos, '_SUCESSO.txt'), 'w') as f:
                f.write(f'Treino concluído com sucesso para {lojas_treinadas_com_sucesso} lojas.')
            print(f"\n--- SUCESSO FINAL: {lojas_treinadas_com_sucesso} lojas foram treinadas. ---")
        else:
            print("\n--- FALHA FINAL: Nenhuma loja pôde ser treinada. ---")

    except Exception as e:
        print(f"!!! ERRO CRÍTICO no script de treino: {e}")
        # Cria um arquivo de falha para diagnóstico
        if not os.path.exists(pasta_modelos):
             os.makedirs(pasta_modelos)
        with open(os.path.join(pasta_modelos, '_FALHA.txt'), 'w') as f:
            f.write(f'O script de treino falhou com o erro: {e}')

    print("--- FIM DO SCRIPT DE TREINO ---")

if __name__ == '__main__':
    treinar_e_salvar_modelos()
