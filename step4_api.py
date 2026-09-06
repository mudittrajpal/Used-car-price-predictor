
import joblib
import numpy as np
from flask import Flask, request, jsonify


print("Loading model artefacts...")

model    = joblib.load("models/model.pkl")
encoders = joblib.load("models/encoders.pkl")
features = joblib.load("models/features.pkl")

print(f"  Model    : loaded")
print(f"  Encoders : {list(encoders.keys())}")
print(f"  Features : {features}")


app = Flask(__name__)


def encode_input(data: dict) -> np.ndarray:
    
    row = []

    for feature in features:
        value = data.get(feature)

        if value is None:
            raise ValueError(f"Missing field: '{feature}'")

        if feature in encoders:
            # Categorical — check the value is known before encoding
            le           = encoders[feature]
            known_values = list(le.classes_)

            if str(value) not in known_values:
                raise ValueError(
                    f"Unknown value '{value}' for '{feature}'. "
                    f"Valid options include: {known_values[:10]}..."
                )

            encoded = le.transform([str(value)])[0]
            row.append(encoded)
        else:
            # Numeric — just cast to float
            try:
                row.append(float(value))
            except (ValueError, TypeError):
                raise ValueError(f"Field '{feature}' must be a number. Got: '{value}'")

    return np.array(row).reshape(1, -1)


@app.route("/health", methods=["GET"])
def health():
    
    return jsonify({
        "status"  : "ok",
        "model"   : "RandomForestRegressor",
        "features": features
    })


@app.route("/predict", methods=["POST"])
def predict():
    
    data = request.get_json()

    if not data:
        return jsonify({
            "status" : "error",
            "message": "Request body must be JSON. Set Content-Type: application/json."
        }), 400

    # Encode the input and run it through the model
    try:
        input_array      = encode_input(data)
        predicted_price  = model.predict(input_array)[0]
        predicted_price  = round(float(predicted_price), 2)

    except ValueError as e:
        return jsonify({
            "status" : "error",
            "message": str(e)
        }), 422

    return jsonify({
        "status"               : "success",
        "predicted_price_euro" : predicted_price,
        "input_received"       : data
    })


@app.route("/valid-values", methods=["GET"])
def valid_values():
    
    field = request.args.get("field")

    if not field:
        # Return all categorical fields
        return jsonify({
            col: list(le.classes_)
            for col, le in encoders.items()
        })

    if field not in encoders:
        return jsonify({
            "status" : "error",
            "message": f"'{field}' is not a categorical field. Categorical fields: {list(encoders.keys())}"
        }), 404

    return jsonify({
        field  : list(encoders[field].classes_),
        "count": len(encoders[field].classes_)
    })


# ── Start the server ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=============================================")
    print("  Used Car Price Predictor — API")
    print("=============================================")
    print("  Server running at: http://127.0.0.1:5000")
    print("\n  Endpoints:")
    print("    GET  /health        — check server is running")
    print("    POST /predict       — get a price prediction")
    print("    GET  /valid-values  — list valid categorical values")
    print("\n  Press Ctrl+C to stop the server.")
    print("=============================================\n")

    # debug=False for cleaner output; set True if you want auto-reload on code changes
    app.run(host="127.0.0.1", port=5000, debug=False)


# ── How to test ───────────────────────────────────────────────────────────────
#
# Open a SECOND terminal while the server is running and try these:
#
# 1. Health check:
#    curl http://127.0.0.1:5000/health
#
# 2. Get a price prediction:
#    curl -X POST http://127.0.0.1:5000/predict \
#         -H "Content-Type: application/json" \
#         -d '{
#               "brand"            : "bmw",
#               "model"            : "BMW 316",
#               "color"            : "black",
#               "year"             : 2018,
#               "power_kw"         : 110,
#               "transmission_type": "Automatic",
#               "fuel_type"        : "Diesel",
#               "mileage_in_km"    : 75000
#             }'
#
# 3. Check valid brands:
#    curl "http://127.0.0.1:5000/valid-values?field=brand"
#
# 4. Check valid fuel types:
#    curl "http://127.0.0.1:5000/valid-values?field=fuel_type"