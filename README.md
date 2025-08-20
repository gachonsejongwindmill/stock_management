
## 1. ngrok 설치

1. **ngrok 다운로드**  
[Windows용 ngrok 다운로드](https://ngrok.com/downloads/windows) 페이지에서 ngrok을 설치합니다.

2. **ngrok 인증키 설정**  
[ngrok Authtoken 발급](https://dashboard.ngrok.com/get-started/your-authtoken) 페이지에서 인증 토큰을 발급받습니다.

3. **ngrok Authtoken 등록**  
`ngrok.exe`를 `start.bat` 파일이 위치한 폴더에 두고, CMD에서 해당 경로로 이동 후 아래 명령어를 실행합니다.

```bash
ngrok.exe config add-authtoken <여기에_복사한_authtoken>
```

---

## 2. Ollama 설치

1. **Ollama 다운로드 및 설치**  
[Ollama 설치 페이지](https://ollama.com/download)에서 설치합니다.

2. **설치 확인**  
CMD에서 아래 명령어로 Ollama가 정상 설치되었는지 확인합니다.

```bash
ollama run gemma3:12b
```

---

## 3. 실행 전 확인 사항

- Anaconda가 설치되어 있는지 확인
- ngrok이 설치되어 있는지 확인
- Ollama가 설치되어 있는지 확인
- CUDA가 설치되어 있는지 확인
- `plots_sp500` 파일이 없는 경우, 생성하는데 **RTX 5070 기준 약 3시간** 소요

---

## 4. 오류 발생 시 대처 방법

1. **가상환경 문제**  
`neuralforecast_timeforecast` 가상환경이 생성되지 않거나 실행되지 않을 경우, 첨부된 가상환경을 사용합니다.

2. **문의**  
추가적인 도움은 편하게 아래 이메일로 문의하세요.  
**good117454@gmail.com**

## 5. 시작 방법

1. 위의 사항을 명심한 후 github에 있는 모든 코드를 복사한 후, cmd로 해당 페이지로 이동한 후, start.bat을 실행하면 됩니다.
2. start.bat을 실행하면 자동으로 neuralforecast_timeforecast 가상환경을 만들것입니다.
