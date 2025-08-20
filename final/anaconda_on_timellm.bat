@echo off

echo Received arguments: %1 %2 %3
set STRING_ARG=%1
set INT_ARG=%2
set INT_ARG2=%3
CALL conda activate neuralforecast_timeforecast
python "C:\Users\good1\Desktop\summer_vacation\windmill\gpu_server\final\timellm.py" %STRING_ARG% %INT_ARG% %INT_ARG2%