def parse_feature_flags(flag_int):
    bin_flag = str(flag_int).zfill(13) 
    print(f"이진 문자열로 변환된 feature flags: {bin_flag}")
    return {
        'volume':        bin_flag[0] == '1',
        'open':          bin_flag[1] == '1',
        'high':          bin_flag[2] == '1',
        'low':           bin_flag[3] == '1',
        'interest_rate': bin_flag[4] == '1',
        'per':          bin_flag[5] == '1',  # 주가수익비율
        'pbr':          bin_flag[6] == '1',  # 주가순자산비율
        'psr':          bin_flag[7] == '1',  # 주가매출비율
        's&p500':      bin_flag[8] == '1',  # S&P 500 지수
        'roe':          bin_flag[9] == '1',  # 자기자본이익률
        'roa':          bin_flag[10] == '1', # 총자산이익률
        'operating_margin': bin_flag[11] == '1',  # 영업이익률
        'net_margin':   bin_flag[12] == '1',  # 순이익률
    }
def main():
    import matplotlib.pyplot as plt
    from transformers import AutoConfig
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TimeLLM
    from neuralforecast.utils import AirPassengersPanel
    from transformers import AutoConfig
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import yfinance as yf
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST
    from neuralforecast.losses.pytorch import DistributionLoss
    from neuralforecast.utils import augment_calendar_df
    import matplotlib as mpl
    from datetime import datetime, date
    import sys
    import json
    import pandas_market_calendars as mcal
    print(sys.argv)
    string_value = sys.argv[1] 

    flag_value = sys.argv[2]

    timeseries_length = sys.argv[3]

    print(flag_value, type(flag_value), "flag_value의 타입")
    feature_flags = parse_feature_flags(flag_value)
    print(f"선택된 feature flags: {feature_flags}")

    today_date = date.today()

    mpl.rcParams['font.family'] = 'Malgun Gothic'

    # 마이너스 기호 깨짐 방지
    mpl.rcParams['axes.unicode_minus'] = False


    ticker = string_value
    start_date = "2021-01-01"
    end_date = today_date


    nasdaq_data = yf.download(ticker, start=start_date, end=end_date, group_by=None)

    ticker_yf = yf.Ticker(ticker)
    info = ticker_yf.info

    balance = ticker_yf.balance_sheet.T
    income = ticker_yf.income_stmt.T
    roe = income["Net Income"] / balance["Stockholders Equity"]
    roa = income["Net Income"] / balance["Total Assets"]
    operating_margin = income["Operating Income"] / income["Total Revenue"]
    net_margin = income["Net Income"] / income["Total Revenue"]
    roe.dropna(axis=0, how='any', inplace=True)
    roa.dropna(axis=0, how='any', inplace=True)
    operating_margin.dropna(axis=0, how='any', inplace=True)
    net_margin.dropna(axis=0, how='any', inplace=True)
    roe = roe.sort_index()
    roa = roa.sort_index()
    operating_margin = operating_margin.sort_index()
    net_margin = net_margin.sort_index()

    sector_string_value = info.get("sector", "no data")
    print("업종:", sector_string_value)
    nasdaq_data.columns = nasdaq_data.columns.get_level_values(1)

    # 데이터 구조 확인
    print(f"{string_value} 데이터 구조:")
    print(nasdaq_data.head())
    print(f"Index type: {type(nasdaq_data.index)}")
    print(f"Columns: {nasdaq_data.columns}")

    nasdaq_data_reset = nasdaq_data.reset_index()


    import pandas_datareader.data as web
    interest_df = web.DataReader('DFEDTARU', 'fred', start_date, end_date)

    interest_df_withdate=interest_df.reset_index()
    nasdaq_dates = nasdaq_data_reset['Date'].unique()
    filtered_interest_df = interest_df_withdate[interest_df_withdate['DATE'].isin(nasdaq_dates)].copy()

    roe_daily = roe.reindex(nasdaq_data_reset['Date'],method='ffill').fillna(method='bfill')
    roa_daily = roa.reindex(nasdaq_data_reset['Date'],method='ffill').fillna(method='bfill')
    operating_margin_daily = operating_margin.reindex(nasdaq_data_reset['Date'],method='ffill').fillna(method='bfill')
    net_margin_daily = net_margin.reindex(nasdaq_data_reset['Date'],method='ffill').fillna(method='bfill')

    forecast_input = {}
    if feature_flags['volume']:
        forecast_input['volume'] = nasdaq_data_reset['Volume']
    if feature_flags['open']:
        forecast_input['open'] = nasdaq_data_reset['Open']
    if feature_flags['high']:
        forecast_input['high'] = nasdaq_data_reset['High']
    if feature_flags['low']:
        forecast_input['low'] = nasdaq_data_reset['Low']
    if feature_flags['interest_rate']:
        forecast_input['interest_rate'] = filtered_interest_df['DFEDTARU'].values
    if feature_flags['roe'] and roe_daily is not None:
        forecast_input['roe'] = roe_daily.values
    if feature_flags['roa'] and roa_daily is not None:
        forecast_input['roa'] = roa_daily.values
    if feature_flags['operating_margin'] and operating_margin_daily is not None:
        forecast_input['operating_margin'] = operating_margin_daily.values
    if feature_flags['net_margin'] and net_margin_daily is not None:
        forecast_input['net_margin'] = net_margin_daily.values

    if feature_flags['s&p500']:
        # S&P 500 지수 데이터 다운로드
        sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
        sp500_data.columns= sp500_data.columns.droplevel(1)
        sp500_data_reset = sp500_data.reset_index()
        sp500_dates = sp500_data_reset['Date'].unique()
        filtered_sp500_df = sp500_data_reset[sp500_data_reset['Date'].isin(nasdaq_dates)].copy()
        forecast_input['s&p500'] = filtered_sp500_df['Close'].values

    nasdaq_df = pd.DataFrame({
        'unique_id': [string_value] * len(nasdaq_data_reset),
        'ds': nasdaq_data_reset['Date'],
        'y': nasdaq_data_reset['Close'],
        **forecast_input 
    })

    nasdaq_df['ds'] = pd.to_datetime(nasdaq_df['ds'])
    nasdaq_df = nasdaq_df[['unique_id', 'ds', 'y'] + list(forecast_input.keys())].copy() 
    nasdaq_df, calendar_cols = augment_calendar_df(df=nasdaq_df, freq='D')
    total_days = len(nasdaq_df)
    test_days = int(timeseries_length)
    train_size = total_days - test_days
    Y_train_df = nasdaq_df.iloc[:train_size].copy()
    Y_train_df['trend'] = np.arange(len(Y_train_df))
    per_value= info.get("forwardPE", "정보없음")
    pbr_value = info.get("priceToBook", "정보없음")
    psr_value = info.get("priceToSales", "정보없음")

    combine_text = ""
    if feature_flags['per'] and per_value != "정보없음":  
        combine_text += f"PER : {per_value} "
    if feature_flags['pbr'] and pbr_value != "정보없음":  
        combine_text += f"PBR : {pbr_value} "
    if feature_flags['psr'] and psr_value != "정보없음":
        combine_text += f"PSR : {psr_value} "
    prompt_prefix = f"This dataset contains daily closing prices of {ticker}. Sector: {sector_string_value}. {combine_text} The data exhibits typical characteristics of financial time series: non stationarity, volatility clustering, trends or cycles, and occasional spikes or drops due to market events. Please analyze patterns and provide forecasting insights."
    print("프롬프트 프리픽스:",type(prompt_prefix),len(prompt_prefix))
    print("prompt_prefix",prompt_prefix[:100])
    timellm = TimeLLM(h=test_days,
                    input_size=test_days*12,
                    llm='openai-community/gpt2',
                    prompt_prefix=prompt_prefix,
                    max_steps=100,
                    batch_size=16,
                    valid_batch_size=16,
                    windows_batch_size=16)

    nf = NeuralForecast(
        models=[timellm],
        freq='D'
    )


    nf.fit(df=Y_train_df, val_size=test_days)
    future_df = nf.make_future_dataframe(df=Y_train_df)
    print(f"생성된 미래 데이터프레임 크기: {len(future_df)}")
    print(f"미래 기간: {future_df['ds'].min()} ~ {future_df['ds'].max()}")
    forecasts = nf.predict(futr_df=future_df)

    nyse = mcal.get_calendar('NYSE')

    start_date = today_date + pd.Timedelta(days=1)
    end_date = start_date + pd.Timedelta(days=1000) 

    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)
    trading_days = trading_days.tz_localize(None)  # 타임존 제거

    future_trading_days = trading_days[:test_days]
    trading_dates_str = future_trading_days.strftime('%Y-%m-%d %H:%M:%S')
    new_dates = pd.to_datetime(trading_dates_str)
    forecasts.loc[:len(new_dates)-1, 'ds'] = new_dates.values

    return forecasts
if __name__ == "__main__":
    import traceback
    try:
        result_df = main()
        result = result_df.to_dict(orient="records")
        import json
        print(json.dumps(result, ensure_ascii=False, default=str))
    except Exception as e:
        print("에러 발생!")
        traceback.print_exc()

