from flask import Blueprint, request, jsonify
from services import generate_advisory

bp = Blueprint("routes", __name__)

@bp.route("/api/advisor", methods=["POST"])
def advisor():
    if not request.is_json:
        return jsonify({"error": "❌ Request must be application/json"}), 415

    data = request.get_json()
    print("📩 Incoming request:", data)

    try:
        response = generate_advisory(data)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
