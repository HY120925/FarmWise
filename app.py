from flask import Flask, render_template, request, jsonify
from Src.predict_full_pipeline import run_pipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        print("Received:", data)

        results = run_pipeline(data)

        return jsonify({"results": results}) 

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)