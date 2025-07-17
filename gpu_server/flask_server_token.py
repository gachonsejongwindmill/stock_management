from flask import Flask, request

app = Flask(__name__)

@app.route("/oauth/callback")
def oauth_callback():
    code = request.args.get('code')
    return f"인가 코드: {code}"

if __name__ == "__main__":
    app.run(port=5000)