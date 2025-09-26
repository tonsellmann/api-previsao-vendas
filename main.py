from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from prophet import Prophet

app = FastAPI()

@app.post("/prever-vendas/")
async def prever_vendas(
    arquivo: UploadFile = File(...),
    lojnumero: str = Form(...),
    ano: int = Form(None),
    mes: int = Form(None),
    prox_12_meses: bool = Form(False)
):
    try:
        df = pd.read_csv(io.BytesIO(await arquivo.read()), sep=';')

        for col in ['LOJNUMERO', 'ANO', 'MES', 'QUANTIDADE', 'SOMA']:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Coluna obrigatória ausente: {col}")

        df_loja = df[df['LOJNUMERO'].astype(str) == str(lojnumero)].copy()
        if df_loja.empty:
            raise HTTPException(status_code=404, detail="Loja não encontrada no arquivo.")

        df_loja['data'] = pd.to_datetime(df_loja['ANO'].astype(str) + '-' + df_loja['MES'].astype(str) + '-01')
        df_loja = df_loja.sort_values('data')

        if len(df_loja) < 13:
            raise HTTPException(status_code=400, detail="São necessários pelo menos 13 meses de dados para previsão.")
        df_loja = df_loja.iloc[:-1]

        # Preparar séries para Prophet
        series_qtd = df_loja.groupby('data')['QUANTIDADE'].sum().reset_index().rename(columns={'data':'ds', 'QUANTIDADE':'y'})
        series_valor = df_loja.groupby('data')['SOMA'].sum().reset_index().rename(columns={'data':'ds', 'SOMA':'y'})

        model_qtd = Prophet(yearly_seasonality=True)
        model_valor = Prophet(yearly_seasonality=True)

        model_qtd.fit(series_qtd)
        model_valor.fit(series_valor)

        if prox_12_meses:
            horizon = 12
            future_qtd = model_qtd.make_future_dataframe(periods=horizon, freq='MS')
            future_valor = model_valor.make_future_dataframe(periods=horizon, freq='MS')
        else:
            if ano is None or mes is None:
                raise HTTPException(status_code=400, detail="Informe ano e mês para previsão específica ou marque prox_12_meses.")
            data_final = series_qtd['ds'].max()
            data_prev = pd.Timestamp(year=ano, month=mes, day=1)
            meses_ate = (data_prev.year - data_final.year) * 12 + (data_prev.month - data_final.month)
            if meses_ate <= 0:
                raise HTTPException(status_code=400, detail="A data de previsão deve ser após o último mês do histórico.")
            
            # Criar datas até o mês solicitado, mas depois pegar só o mês desejado
            future_qtd = model_qtd.make_future_dataframe(periods=meses_ate, freq='MS')
            future_valor = model_valor.make_future_dataframe(periods=meses_ate, freq='MS')

            future_qtd = future_qtd.tail(1)
            future_valor = future_valor.tail(1)

        forecast_qtd = model_qtd.predict(future_qtd)
        forecast_valor = model_valor.predict(future_valor)

        # Histórico últimos 12 meses
        historico = []
        ultimos_12_qtd = series_qtd.tail(12).set_index('ds')
        ultimos_12_val = series_valor.tail(12).set_index('ds')
        for dt in ultimos_12_qtd.index:
            historico.append({
                "data": dt.strftime('%Y-%m-%d'),
                "ano": dt.year,
                "mes": dt.month,
                "quantidade": int(ultimos_12_qtd.loc[dt]['y']),
                "soma": float(ultimos_12_val.loc[dt]['y'])
            })

        # Previsão (mês único ou 12 meses)
        if prox_12_meses:
            previsoes = []
            previsao_qtd_slice = forecast_qtd.tail(horizon).set_index('ds')
            previsao_valor_slice = forecast_valor.tail(horizon).set_index('ds')
            for dt in previsao_qtd_slice.index:
                qtd = max(0, round(previsao_qtd_slice.loc[dt]['yhat']))
                val = max(0, round(previsao_valor_slice.loc[dt]['yhat'], 2))
                previsoes.append({
                    "data_prevista": dt.strftime('%Y-%m-%d'),
                    "ano": dt.year,
                    "mes": dt.month,
                    "quantidade_prevista": int(qtd),
                    "soma_prevista": float(val)
                })
        else:
            dt = forecast_qtd['ds'].iloc[-1]
            qtd = max(0, round(forecast_qtd['yhat'].iloc[-1]))
            val = max(0, round(forecast_valor['yhat'].iloc[-1], 2))
            previsoes = [{
                "data_prevista": dt.strftime('%Y-%m-%d'),
                "ano": dt.year,
                "mes": dt.month,
                "quantidade_prevista": int(qtd),
                "soma_prevista": float(val)
            }]

        # Gráfico
        plt.figure(figsize=(12,6))

        plt.subplot(2,1,1)
        plt.plot(series_qtd['ds'], series_qtd['y'], label='Histórico (quantidade)')
        plt.plot(forecast_qtd['ds'], forecast_qtd['yhat'], label='Previsão (quantidade)')
        plt.title(f'Previsão de QUANTIDADE - Loja {lojnumero}')
        plt.xlabel('Data')
        plt.ylabel('Quantidade')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(series_valor['ds'], series_valor['y'], label='Histórico (soma)')
        plt.plot(forecast_valor['ds'], forecast_valor['yhat'], label='Previsão (soma)')
        plt.title(f'Previsão de SOMA - Loja {lojnumero}')
        plt.xlabel('Data')
        plt.ylabel('Valor (SOMA)')
        plt.legend()

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return JSONResponse(content={
            "historico_ultimos_12": historico,
            "previsoes": previsoes,
            "grafico_base64": grafico_base64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro inesperado: {e}")
