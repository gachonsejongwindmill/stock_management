# time llm 예측 volume, interest rate 추가 한번만 돌림, feature 추가전이 성능이 보편적으로 좋음
def parse_feature_flags(flag_int):
    # 5자리 정수 → 이진 문자열로 변환 후 앞에서부터 읽기
    bin_flag = str(flag_int).zfill(5) # 일단 5자리로 고정
    print(f"이진 문자열로 변환된 feature flags: {bin_flag}")
    return {
        'volume':        bin_flag[0] == '1',
        'open':          bin_flag[1] == '1',
        'high':          bin_flag[2] == '1',
        'low':           bin_flag[3] == '1',
        'interest_rate': bin_flag[4] == '1',
    }
def main():
    import matplotlib.pyplot as plt
    from transformers import AutoConfig
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TimeLLM
    from transformers import AutoConfig
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import yfinance as yf
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import DistributionLoss
    from neuralforecast.utils import augment_calendar_df
    import matplotlib as mpl
    from datetime import datetime, date
    import sys
    import json
    #if len(sys.argv) < 3:
    #    print("[]")
    #    return
    #print("hello")
    print(sys.argv)
    string_value = sys.argv[1]
    flag_value = sys.argv[2]
    feature_flags = parse_feature_flags(flag_value)
    print(f"선택된 feature flags: {feature_flags}")

    today_date = date.today()
    # 맑은 고딕으로 폰트 설정 (Windows 기본 폰트 이름 그대로 사용)
    mpl.rcParams['font.family'] = 'Malgun Gothic'

    # 마이너스 기호 깨짐 방지
    mpl.rcParams['axes.unicode_minus'] = False
    # patchtst 일단위로 close로만 예측측

    # NASDAQ 데이터 다운로드 (2015-04-30 ~ 2025-05-30)
    #ticker = "^IXIC"
    ticker = string_value
    start_date = "2015-04-30"
    end_date = today_date

    # MultiIndex 컬럼을 방지하기 위해 단일 티커로 다운로드
    nasdaq_data = yf.download(ticker, start=start_date, end=end_date, group_by=None)
    nasdaq_data.columns = nasdaq_data.columns.get_level_values(1)

    # 데이터 구조 확인
    print(f"{string_value} 데이터 구조:")
    print(nasdaq_data.head())
    print(f"Index type: {type(nasdaq_data.index)}")
    print(f"Columns: {nasdaq_data.columns}")

    # 데이터프레임 리셋 후 변환
    nasdaq_data_reset = nasdaq_data.reset_index()

    # NeuralForecast 형식으로 데이터 변환 (일별 데이터 유지)

    import pandas_datareader.data as web
    interest_df = web.DataReader('DFEDTARU', 'fred', start_date, end_date)

    interest_df_withdate=interest_df.reset_index()
    nasdaq_dates = nasdaq_data_reset['Date'].unique()
    filtered_interest_df = interest_df_withdate[interest_df_withdate['DATE'].isin(nasdaq_dates)].copy()

    #########################################
    # feature_flags에 따라 예측 입력 데이터 구성 
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

    nasdaq_df = pd.DataFrame({
        'unique_id': [string_value] * len(nasdaq_data_reset),
        'ds': nasdaq_data_reset['Date'],
        'y': nasdaq_data_reset['Close'],
        **forecast_input
    })

    nasdaq_df['ds'] = pd.to_datetime(nasdaq_df['ds'])
    nasdaq_df = nasdaq_df[['unique_id', 'ds', 'y'] + list(forecast_input.keys())].copy() 

        # 캘린더 특성 추가 (일별 빈도)
    nasdaq_df, calendar_cols = augment_calendar_df(df=nasdaq_df, freq='D')

    # Train/Test 분할 (마지막 30일을 테스트 세트로)
    total_days = len(nasdaq_df)
    test_days = 30  # 30일 예측
    train_size = total_days - test_days

    Y_train_df = nasdaq_df.iloc[:train_size].copy()
    Y_test_df = nasdaq_df.iloc[train_size:].reset_index(drop=True)
    Y_train_df['trend'] = np.arange(len(Y_train_df))
    Y_test_df['trend'] = np.arange(len(Y_train_df), len(Y_train_df) + len(Y_test_df))
    print(f"전체 데이터 포인트: {total_days}일")
    print(f"훈련 데이터: {len(Y_train_df)}일")
    print(f"테스트 데이터: {len(Y_test_df)}일")
    print(f"훈련 기간: {Y_train_df['ds'].min()} ~ {Y_train_df['ds'].max()}")
    print(f"테스트 기간: {Y_test_df['ds'].min()} ~ {Y_test_df['ds'].max()}")


    prompt_prefix = f"This dataset contains daily closing prices of the {string_value} index. The data exhibits typical characteristics of financial time series, such as non-stationarity, volatility clustering, and potential trends or cycles. It may also include sudden spikes or drops due to market events. Please analyze this dataset to identify underlying patterns, trends, and any anomalies, and provide insights that could be useful for forecasting future values."

    timellm = TimeLLM(h=30,
                    input_size=365,
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


    nf.fit(df=Y_train_df, val_size=60)
    future_df = nf.make_future_dataframe(df=Y_train_df)
    print(f"생성된 미래 데이터프레임 크기: {len(future_df)}")
    print(f"미래 기간: {future_df['ds'].min()} ~ {future_df['ds'].max()}")
    forecasts = nf.predict(futr_df=future_df)

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))


    # 실제값과 예측값 매칭
    matching_dates = []
    actual_values = []
    for pred_date in forecasts['ds'].values:
        if pred_date in Y_test_df['ds'].values:
            matching_idx = Y_test_df[Y_test_df['ds'] == pred_date].index
            if len(matching_idx) > 0:
                matching_dates.append(pred_date)
                # 값만 추출
                actual_values.append(Y_test_df.iloc[matching_idx[0]]['y'])

    if len(matching_dates) > 0:
        plot_test_df = pd.DataFrame({
            'unique_id': [string_value] * len(matching_dates), # possible error here
            'ds': matching_dates,
            'y': actual_values
        })
        pred_matching = forecasts[forecasts['ds'].isin(matching_dates)].reset_index(drop=True)
        Y_hat_matched = pred_matching.drop(columns=['unique_id','ds'])
        plot_df = pd.concat([plot_test_df, Y_hat_matched], axis=1)
    else:
        plot_df = forecasts.copy()
        plot_df['y'] = None

    recent_train = Y_train_df.tail(90)
    plot_df_combined = pd.concat([recent_train, plot_df], ignore_index=True)

    # 시각화
    plt.figure(figsize=(15, 8))
    plot_data = plot_df_combined[plot_df_combined.unique_id==string_value].copy() #possible error here
    train_data = plot_data[plot_data['ds'] <= Y_train_df['ds'].max()]
    pred_data = plot_data[plot_data['ds'] > Y_train_df['ds'].max()]

    plt.plot(train_data['ds'], train_data['y'], c='black', label='훈련 데이터')
    if len(pred_data) > 0:
        if 'y' in pred_data.columns:
            valid_actual = pred_data.dropna(subset=['y'])
            if len(valid_actual) > 0:
                plt.plot(valid_actual['ds'], valid_actual['y'], c='red', label='실제값', marker='o')
        if 'TimeLLM' in pred_data.columns:
            plt.plot(pred_data['ds'], pred_data['TimeLLM'], c='blue', label='예측값')
    if len(Y_train_df) > 0:
        plt.axvline(x=Y_train_df['ds'].max(), color='green', linestyle='--', alpha=0.7, label='예측 시작')

    plt.title(f'{string_value} Composite Index - TimeLLM 일별 예측', fontsize=16)
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('지수값', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.show()

    # 예측 성능 평가
    if len(matching_dates) > 0:
        y_true = np.array(actual_values)
        y_pred = pred_matching['TimeLLM'].values
        mse_score = mse(y_true, y_pred)
        mae_score = mae(y_true, y_pred)
        mape_score = 100 * mae_score / y_true.mean()
        print(f"\n예측 성능 (매칭된 {len(matching_dates)}일):")
        print(f"MSE: {mse_score:.2f}")
        print(f"MAE: {mae_score:.2f}")
        print(f"MAPE: {mape_score:.2f}%")
        comparison_df = pd.DataFrame({
            '날짜': matching_dates,
            '실제값': y_true,
            '예측값': y_pred,
            '오차': y_true - y_pred,
            '절대오차율(%)': 100 * abs(y_true - y_pred) / y_true
        })
        print(f"\n예측 결과 비교:")
        print(comparison_df.to_string(index=False, float_format='%.2f'))
    else:
        print(f"예측된 기간: {forecasts['ds'].min()} ~ {forecasts['ds'].max()}")

    print(f"\n예측 결과 (첫 10일):")
    forecast_display = forecasts.head(10)[['ds', 'TimeLLM']]
    print(forecast_display.to_string(index=False, float_format='%.2f'))
    return forecasts
if __name__ == "__main__":
    result_df = main()
    result = result_df.to_dict(orient="records")
    import json
    print(json.dumps(result, ensure_ascii=False, default=str))

