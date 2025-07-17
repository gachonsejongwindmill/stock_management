#!/bin/bash

while true
do
    echo "Starting ngrok..."
    ./ngrok http 5000 > ngrok.log &
    NGROK_PID=$!

    sleep 5  # ngrok이 URL을 생성할 시간을 줌

    # ngrok의 public URL 추출
    NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | grep -o 'https://[0-9a-z]*\.ngrok-free\.app')

    echo "ngrok URL: $NGROK_URL"

    # 카카오톡 알림 전송
    python send_kakao.py "$NGROK_URL"

    # ngrok 프로세스를 감시
    wait $NGROK_PID

    echo "ngrok crashed or exited. Restarting..."
    sleep 2
done
