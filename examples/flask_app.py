"""
Example: Use ProgramAsWeights in a Flask API.

Define functions in English, run them locally as API endpoints.

Usage:
    pip install programasweights flask
    python flask_app.py

Then:
    curl -X POST http://localhost:5000/triage \
         -H "Content-Type: application/json" \
         -d '{"text": "Urgent: the server is down!"}'
"""
from flask import Flask, request, jsonify
import programasweights as paw

app = Flask(__name__)

triage = paw.function("email-triage")
json_fixer = paw.function(
    paw.compile(
        "Fix malformed JSON: repair missing quotes and trailing commas",
        compiler="paw-4b-qwen3-0.6b",
    ).id
)


@app.route("/triage", methods=["POST"])
def triage_message():
    text = request.json.get("text", "")
    result = triage(text)
    return jsonify({"urgency": result})


@app.route("/fix-json", methods=["POST"])
def fix_json():
    text = request.json.get("text", "")
    result = json_fixer(text)
    return jsonify({"fixed": result})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Starting ProgramAsWeights Flask API...")
    print("  POST /triage    - Classify message urgency")
    print("  POST /fix-json  - Repair malformed JSON")
    app.run(host="0.0.0.0", port=5000)
