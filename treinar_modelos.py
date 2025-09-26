# treinar_modelos.py (Versão Otimizada para Baixa Memória)

import pandas as pd
import statsmodels.api as sm
import numpy as np
import pickle
import os
import shutil

def treinar_e_salvar_modelos():
    print("--- INÍCIO DO SCRIPT DE TREINO (MODO ECONÓMICO) ---")
    arquivo_csv = 'vendas.csv'
    pasta_modelos = 'modelos_salvos'

    if os.path.exists(pasta_modelos):
        shutil.rmtree(pasta_modelos)
    os.makedirs(pasta_modelos)

    try:
        print(f"A carregar dados de '{arquivo_csv}' de forma otimizada...")
        df_completo = pd.read_csv(arquivo_csv, delimiter=';')
        lojas_unicas = df_completo['LOJNUMERO'].unique()
        print(f"Encontradas {len(lojas_unicas)} lojas. A iniciar treino sequencial...")

        lojas_treinadas = 0
        for loja_id in lojas_unicas:
            try:
                # Isola os dados de uma única loja para poupar memória
                df_loja_temp = df_completo[df_completo['LOJNUMERO'] == loja_id].copy()
                
                if len(df_loja_temp) < 12:
                    print(f"  - Loja {loja_id}: Ignorada (dados insuficientes).")
                    continue

                print(f"  -> A processar e treinar Loja {loja_id}...")
                
                df_loja_temp['DATA'] = pd.to_datetime(df_loja_temp['ANO'].astype(str) + '-' + df_loja_temp['MES'].astype(str) + '-01')
                df_loja_temp = df_loja_temp.set_index('DATA').sort_index()

                df_loja_temp['SOMA_LOG'] = np.log1p(df_loja_temp['SOMA'])
                
                modelo_soma = sm.tsa.SARIMAX(df_loja_temp['SOMA_LOG'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
                
                with open(f'{pasta_modelos}/modelo_loja_{loja_id}_soma.pkl', 'wb') as pkl:
                    pickle.dump(modelo_soma, pkl)
                
                print(f"     Modelo para Loja {loja_id} salvo com sucesso.")
                lojas_treinadas += 1

            except Exception as e:
                print(f"    !!! ERRO ao treinar para a Loja {loja_id}: {e}")
            finally:
                # Limpa a memória antes de passar para a próxima loja
                del df_loja_temp

        # Se pelo menos uma loja foi treinada, o processo é um sucesso
        if lojas_treinadas > 0:
            with open(os.path.join(pasta_modelos, '_SUCESSO.txt'), 'w') as f:
                f.write(f'Treino concluído com sucesso para {lojas_treinadas} lojas.')
            print(f"\n--- SUCESSO: {lojas_treinadas} lojas foram treinadas. ---")
        else:
            print("\n--- FALHA: Nenhuma loja pôde ser treinada. ---")

    except Exception as e:
        print(f"!!! ERRO CRÍTICO no script de treino: {e}")
    
    print("--- FIM DO SCRIPT DE TREINO ---")

if __name__ == '__main__':
    treinar_e_salvar_modelos()
