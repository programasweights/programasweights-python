"""
Example: Use ProgramAsWeights in a Flask API.

Replace expensive OpenAI API calls with local neural programs.

Usage:
    pip install programasweights flask
    python flask_app.py

Then:
    curl -X POST http://localhost:5000/extract-emails \
         -H "Content-Type: application/json" \
         -d '{"text": "Contact alice@company.com or bob@example.org"}'
"""
from flask import Flask, request, jsonify
import programasweights as paw

app = Flask(__name__)

# Load programs once at startup (cached, fast after first call)
email_extractor = paw.function("programasweights/email-extractor")
sentiment_analyzer = paw.function("programasweights/sentiment-classifier")


@app.route("/extract-emails", methods=["POST"])
def extract_emails():
    text = request.json.get("text", "")
    result = email_extractor(text)
    return jsonify({"emails": result})


@app.route("/sentiment", methods=["POST"])
def sentiment():
    text = request.json.get("text", "")
    result = sentiment_analyzer(text)
    return jsonify({"sentiment": result})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Starting ProgramAsWeights Flask API...")
    print("  POST /extract-emails  - Extract emails from text")
    print("  POST /sentiment       - Analyze sentiment")
    app.run(host="0.0.0.0", port=5000)
