# treinar_modelos.py (Versão Final e Robusta)

import pandas as pd
import statsmodels.api as sm
import numpy as np
import pickle
import os
import shutil

def treinar_e_salvar_modelos():
    """
    Executa o treino de forma robusta, otimizada para ambientes com pouca memória.
    """
    print("--- INÍCIO DO PROCESSO DE TREINO AUTOMÁTICO ---")
    arquivo_csv = 'vendas.csv'
    pasta_modelos = 'modelos_salvos'

    if not os.path.exists(arquivo_csv):
        print(f"!!! ERRO FATAL: O arquivo '{arquivo_csv}' não foi encontrado. O treino não pode continuar.")
        return

    if os.path.exists(pasta_modelos):
        shutil.rmtree(pasta_modelos)
    os.makedirs(pasta_modelos)

    try:
        print("A identificar lojas únicas de forma eficiente...")
        lojas_unicas = pd.read_csv(arquivo_csv, delimiter=';', usecols=['LOJNUMERO'], dtype={'LOJNUMERO': int})['LOJNUMERO'].unique()
        print(f"Encontradas {len(lojas_unicas)} lojas. A iniciar treino sequencial...")

        lojas_treinadas_com_sucesso = 0
        for loja_id in lojas_unicas:
            try:
                print(f"  -> A processar Loja {loja_id}...")
                
                df_loja = pd.read_csv(arquivo_csv, delimiter=';', dtype={'LOJNUMERO': int})
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
                print(f"    !!! AVISO: Falha ao treinar para a Loja {loja_id}: {e}")
            finally:
                if 'df_loja' in locals():
                    del df_loja
        
        if lojas_treinadas_com_sucesso > 0:
            with open(os.path.join(pasta_modelos, '_SUCESSO.txt'), 'w') as f:
                f.write(f'Treino concluído com sucesso para {lojas_treinadas_com_sucesso} lojas.')
            print(f"\n--- SUCESSO: {lojas_treinadas_com_sucesso} lojas foram treinadas. ---")
        else:
            raise Exception("Nenhuma loja pôde ser treinada.")

    except Exception as e:
        print(f"!!! ERRO FATAL no script de treino: {e}")
        # Se o treino falhar, o processo de build irá parar com um erro claro.
        raise e

    print("--- FIM DO PROCESSO DE TREINO ---")

if __name__ == '__main__':
    treinar_e_salvar_modelos()
