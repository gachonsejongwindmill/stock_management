@echo off
chcp 65001 > nul

echo Received arguments: %1 %2
set INT_ARG=%1
set INT_ARG2=%2
CALL conda activate neuralforecast_timeforecast
python "C:\Users\good1\Desktop\summer_vacation\windmill\gpu_server\final\portfolio.py" %INT_ARG% %INT_ARG2%