import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
from datetime import date
import sys
from copy import deepcopy
from neuralforecast import NeuralForecast
from neuralforecast.models import TimeLLM
from neuralforecast.utils import augment_calendar_df
import matplotlib as mpl

# 로그 저장 설정
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", ".log")
sys.stdout = open(log_file, "a", encoding="utf-8")
sys.stderr = open(log_file, "a", encoding="utf-8")

results_path = "results_sp500.csv"
if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)
    existing_tickers = set(results_df['ticker'].astype(str))
    results = results_df.to_dict('records')
else:
    results = []
    existing_tickers = set()

# 설정
features = ['open', 'high', 'low', 'interest_rate', 'roe', 'roa']
start_date = "2021-01-01"
end_date = date.today()
test_days = 30
input_size = 365
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False
os.makedirs("plots_sp500", exist_ok=True)

# S&P500 구성 종목 가져오기 
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
feature_flags = {
    'volume': True,
    'open': False,
    'high': True,
    'low': False,
    'interest_rate': True,
    'roe': False,
    'roa': True,
    'operating_margin': False,
    'net_margin': True,
    's&p500': True,
    'per': True,
    'pbr': True,
    'psr': True
}

# 평가지표
def mse(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))

def process_ticker(ticker):
    try:
        # ------------------- 주가 데이터 -------------------
        df = yf.download(ticker, start=start_date, end=end_date).reset_index()
        df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]
        if df.empty:
            print(f"{ticker}: 데이터 없음, 건너뜀")
            return None

        df = df[['Date', 'Close', 'High','Volume']].copy()
        df['unique_id'] = ticker
        df['ds'] = pd.to_datetime(df['Date'])
        df['y'] = df['Close']

        # ------------------- 재무제표 -------------------
        ticker_yf = yf.Ticker(ticker)
        info = ticker_yf.info
        balance = ticker_yf.balance_sheet.T
        income = ticker_yf.income_stmt.T

        #roe = (income["Net Income"] / balance["Stockholders Equity"]).dropna().sort_index()
        roa = (income["Net Income"] / balance["Total Assets"]).dropna().sort_index()
        if "Operating Income" in income and "Total Revenue" in income:
            operating_margin = (income["Operating Income"] / income["Total Revenue"]).dropna().sort_index()
            df['operating_margin'] = operating_margin.reindex(df['ds'], method='ffill').fillna(method='bfill').reindex(df['ds']).values
        else:
            operating_margin = pd.Series()

        # Net Margin (예외처리)
        if "Net Income" in income and "Total Revenue" in income:
            net_margin = (income["Net Income"] / income["Total Revenue"]).dropna().sort_index()
            df['net_margin'] = net_margin.reindex(df['ds'], method='ffill').fillna(method='bfill').reindex(df['ds']).values
        else:
            net_margin = pd.Series()

        #df['roe'] = roe.reindex(df['ds'], method='ffill').fillna(method='bfill').values
        df['roa'] = roa.reindex(df['ds'], method='ffill').fillna(method='bfill').values
        #df['operating_margin'] = operating_margin.reindex(df['ds'], method='ffill').fillna(method='bfill').values
        #df['net_margin'] = net_margin.reindex(df['ds'], method='ffill').fillna(method='bfill').values

        # ------------------- 금리 -------------------
        if feature_flags['interest_rate']:
            interest_df = web.DataReader('DFEDTARU', 'fred', start_date, end_date).reset_index()
            df = df.merge(interest_df.rename(columns={'DATE': 'ds', 'DFEDTARU': 'interest_rate'}),
                          on='ds', how='left')
            df['interest_rate'].fillna(method='ffill', inplace=True)

        # ------------------- S&P500 -------------------
        if feature_flags['s&p500']:
            sp500 = yf.download('^GSPC', start=start_date, end=end_date).reset_index()
            sp500 = sp500[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 's&p500'})
            sp500['ds'] = pd.to_datetime(sp500['ds'])
            df = df.merge(sp500, on='ds', how='left')
            df['s&p500'].fillna(method='ffill', inplace=True)

        # ------------------- 달력 피처 -------------------
        df = df.drop(columns=['Date', 'Close'])
        df, _ = augment_calendar_df(df=df, freq='D')

        # ------------------- Train/Test Split -------------------
        total_days = len(df)
        train_size = total_days - test_days
        Y_train_df = df.iloc[:train_size].copy()
        Y_test_df = df.iloc[train_size:].reset_index(drop=True)

        Y_train_df['trend'] = np.arange(len(Y_train_df))
        Y_test_df['trend'] = np.arange(len(Y_train_df), len(Y_train_df) + len(Y_test_df))

        # ------------------- Prompt -------------------
        sector = info.get("sector", "N/A")
        per_value = info.get("forwardPE", "N/A")
        pbr_value = info.get("priceToBook", "N/A")
        psr_value = info.get("priceToSales", "N/A")

        combine_text = ""
        if feature_flags['per'] and per_value != "N/A": combine_text += f"PER: {per_value} "
        if feature_flags['pbr'] and pbr_value != "N/A": combine_text += f"PBR: {pbr_value} "
        if feature_flags['psr'] and psr_value != "N/A": combine_text += f"PSR: {psr_value} "

        prompt_prefix = (
            f"This dataset contains daily closing prices of {ticker}. "
            f"Sector: {sector}. {combine_text}"
            "The data exhibits characteristics of financial time series: non-stationarity, volatility clustering, and trends."
        )

        # ------------------- 모델 학습 -------------------
        timellm = TimeLLM(
            h=test_days,
            input_size=input_size,
            llm='openai-community/gpt2',
            prompt_prefix=prompt_prefix,
            max_steps=100,
            batch_size=16,
            valid_batch_size=16,
            windows_batch_size=16
        )
        nf = NeuralForecast(models=[timellm], freq='D')
        nf.fit(df=Y_train_df, val_size=test_days*2)
        forecasts = nf.predict(futr_df=nf.make_future_dataframe(df=Y_train_df))

        # ------------------- 평가 -------------------
        match_dates, actuals = [], []
        for d in forecasts['ds']:
            if d in Y_test_df['ds'].values:
                #actual = Y_test_df[Y_test_df['ds'] == d]['y'].values[0]
                match_dates.append(d)
                #actuals.append(actual)

        if not match_dates:
            print(f"{ticker}: 예측값과 실제값 매칭 실패")
            return None

        #y_true = np.array(actuals)
        y_pred = forecasts[forecasts['ds'].isin(match_dates)]['TimeLLM'].values

        forecast_output = pd.DataFrame({
            'Date': match_dates,
            'Predicted': y_pred
        })
        forecast_output.to_csv(f'./plots_sp500/{ticker}_forecast.csv', index=False)
        return {
            'ticker': ticker,
            'features': str(list(df.columns))
        }

    except Exception as e:
        print(f"{ticker}: 에러 발생 - {e}")
        return None


# -----------------------------
# 실행 예시
# -----------------------------
for idx, ticker in enumerate(sp500_tickers, 1):
    if ticker in existing_tickers:
        print(f"⏩ 이미 완료된 종목 건너뜀: {ticker}")
        continue
    print(f"\n== 실험 {idx}/{len(sp500_tickers)}: 종목 = {ticker} ==")
    result = process_ticker(ticker)
    if result:
        results.append(result)



