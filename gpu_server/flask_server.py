from flask import Flask, jsonify, request
import subprocess
import json
app = Flask(__name__)

@app.route('/run-forecast', methods=['POST'])
def run_forecast():
    try:
        request_data = request.get_json()
        print("🔍 Raw Request JSON:", request_data)
        string_value = request_data.get("string_value", "")
        int_value = request_data.get("int_value", "")
        result = subprocess.run(
            [r"C:\Users\good1\Desktop\summer_vacation\windmill\gpu_server\anaconda_on.bat",string_value,str(int_value)],
            capture_output=True,
            text=True,
            shell=True
        )

        print("STDOUT:", result.stdout)
        output = result.stdout
        start_idx = output.find("[{")
        end_idx = output.rfind("}]") + 2
        json_string = output[start_idx:end_idx]
        data = json.loads(json_string)
        print(data)
        #print("STDERR:", result.stderr)
        #print("hello")
        if result.returncode != 0:
            return jsonify({'error': f'Process failed: {result.stderr}'}), 500
        #print(repr(result.stdout))
        #forecast = json.loads(data)
        print("🔮 Forecast:", data)  # ← 콘솔에 결과 출력
        return jsonify({'forecast': data})  # ← 클라이언트에 응답

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

