@echo off
chcp 65001
cd /d %~dp0

REM === Step 0. Conda 환경 확인 및 생성 ===
conda info --envs | findstr /C:"neuralforecast_timeforecast" >nul
if %ERRORLEVEL% NEQ 0 (
    echo 가상환경 neuralforecast_timeforecast가 없습니다. environment.yml을 이용하여 생성합니다.
    conda env create -f environment.yml
)

REM === Step 0.5. 가상환경 활성화 ===
call conda activate neuralforecast_timeforecast

REM === Step 1. plots_sp500 디렉토리 확인 ===
if not exist plots_sp500 (
    echo 포트폴리오에 사용할 데이터가 없어 제작합니다. 대략 3시간 걸립니다.
    python to_make_portfolio.py
)

REM === Step 2. 기존 ngrok 종료 ===
taskkill /F /IM ngrok.exe >nul 2>&1

REM === Step 3. ngrok 실행 ===
set NGROK_EXE=ngrok.exe
start "" /b %NGROK_EXE% http 5001 > ngrok.log
echo ngrok 실행 중... URL 생성 대기

REM === Step 4. ngrok URL 추출 ===
set NGROK_URL=
set /a MAX_TRIES=20
set /a TRY_COUNT=0

:WAIT_NGROK
timeout /t 1 >nul
for /f "tokens=*" %%i in ('powershell -Command "(Invoke-RestMethod http://127.0.0.1:4040/api/tunnels).tunnels[0].public_url"') do set NGROK_URL=%%i
set NGROK_URL=%NGROK_URL: =%
set /a TRY_COUNT+=1
if "%NGROK_URL%"=="" if %TRY_COUNT% LSS %MAX_TRIES% goto WAIT_NGROK

if "%NGROK_URL%"=="" (
    echo ngrok URL 생성 실패. 수동 확인 필요.
) else (
    echo 서버와 AI 서버를 연결시켜주는 주소는 %NGROK_URL%
    python send_kakao.py "%NGROK_URL%" || (
        echo 카카오톡으로 송신을 실패했습니다.
        echo 여기서 서버 주소를 확인해주세요: %NGROK_URL%
    )
)

REM === Step 5. 포트 5001 사용 중인 프로세스 확인 및 종료 ===
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5001" ^| findstr "LISTENING"') do (
    echo 포트 5001을 사용 중인 프로세스 PID: %%a 종료
    taskkill /F /PID %%a >nul 2>&1
)

REM === Step 6. Flask 서버 실행 및 준비 확인 ===
start "" python flask_combine.py
echo AI 서버 부팅 중... 포트 5001 확인

set /a MAX_PORT_TRIES=30
set /a PORT_TRY_COUNT=0

:WAIT_PORT
timeout /t 2 >nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5001" ^| findstr "LISTENING"') do set PORT_OPEN=1
if "%PORT_OPEN%"=="1" (
    echo 포트 5001이 열렸습니다. Flask 서버 실행 완료
    goto CURL_REQUEST
)
set /a PORT_TRY_COUNT+=1
if %PORT_TRY_COUNT% LSS %MAX_PORT_TRIES% goto WAIT_PORT
echo 포트 5001이 열리지 않았습니다. 서버 상태 확인 필요.
goto END

:CURL_REQUEST
REM === Step 7. curl POST 요청 ===
echo 테스트용 POST 요청 전송 중...
curl -X POST "%NGROK_URL%/run-forecast" -H "Content-Type: application/json" -d "{\"string_value\":\"AAPL\", \"int_value1\":1111111111111,  \"int_value2\":30}"
if %ERRORLEVEL%==0 (
    echo 테스트까지 완료되었습니다.
) else (
    echo POST 요청 실패. 서버 주소 및 상태 확인 필요.
)

:END
pause
