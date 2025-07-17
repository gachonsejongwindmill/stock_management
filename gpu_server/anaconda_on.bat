@echo off
:: 첫 번째, 두 번째 인자 받기
print yes
echo Received arguments: %1 %2
set STRING_ARG=%1
set INT_ARG=%2
CALL conda activate neuralforecast_timeforecast
python "C:\Users\good1\Desktop\summer_vacation\windmill\gpu_Server\baselinemodel.py" %STRING_ARG% %INT_ARG%