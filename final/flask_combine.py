from flask import Flask, jsonify, request, Response
import subprocess
import json
import re

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False 
def run_subprocess(command, use_shell=True):

    result = subprocess.run(command, capture_output=True, text=True, shell=use_shell)
    return result.returncode, result.stdout, result.stderr

@app.route('/run-predict', methods=['POST'])
def run_predict():
    try:
        request_data = request.get_json()
        print("🔍 /run-predict Raw Request JSON:", request_data)

        int_value = request_data.get("int_value1", "")
        int_value2 = request_data.get("int_value2", "")

        cmd = rf'"C:\Users\good1\Desktop\summer_vacation\windmill\gpu_server\final\anaconda_on_portfolio.bat" {int_value} {int_value2}'
        rc, stdout, stderr = run_subprocess(cmd, use_shell=True)

        if rc != 0:
            return jsonify({'error': 'Subprocess failed', 'stderr': stderr}), 500
        if isinstance(stdout, bytes):
            stdout = stdout.decode('utf-8')
        print(stdout)
        start_idx = stdout.find('{')
        end_idx = stdout.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            return jsonify({'error': 'Could not find JSON in subprocess output', 'raw_output': stdout}), 500

        json_string = stdout[start_idx:end_idx]
        data = json.loads(json_string)

        print(json.dumps(data, ensure_ascii=False, indent=2))
        json_string2 = json.dumps(data, ensure_ascii=False, indent=2)
        return Response(json_string2, content_type='application/json; charset=utf-8')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/run-forecast', methods=['POST'])
def run_forecast_with_string():
    try:
        request_data = request.get_json()
        print("🔍 /run-forecast Raw Request JSON:", request_data)

        string_value = request_data.get("string_value", "")
        int_value = request_data.get("int_value1", "")
        int_value2 = request_data.get("int_value2", "")

        cmd = rf'"C:\Users\good1\Desktop\summer_vacation\windmill\gpu_server\final\anaconda_on_timellm.bat" {string_value} {int_value} {int_value2}'
        rc, stdout, stderr = run_subprocess(cmd, use_shell=True)

        if rc != 0:
            return jsonify({'error': 'Subprocess failed', 'stderr': stderr}), 500

        start_idx = stdout.find("[{")
        end_idx = stdout.rfind("}]") + 2
        if start_idx == -1 or end_idx == 1:
            m = re.search(r'(\[\s*\{.*\}\s*\])', stdout, re.DOTALL)
            if not m:
                return jsonify({'error': 'Could not find JSON array in subprocess output', 'raw_output': stdout}), 500
            json_string = m.group(1)
        else:
            json_string = stdout[start_idx:end_idx]

        data = json.loads(json_string)
        for item in data:
            if 'TimeLLM' in item and isinstance(item['TimeLLM'], (float, int)):
                item['TimeLLM'] = round(item['TimeLLM'], 2)

        print("🔮 Forecast:", data)
        return jsonify({'forecast': data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
