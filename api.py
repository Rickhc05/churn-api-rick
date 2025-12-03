from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------
# Inicializar API
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------------
# Cargar modelo y scaler entrenados
# -------------------------------------------------
model = joblib.load("modelo_xgb.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------------------------
# Reconstruir el preprocesamiento EXACTO del notebook
# -------------------------------------------------
df = pd.read_excel("Customer_Churn_Dataset.xlsx")
df_clean = df.drop("customerID", axis=1).copy()

df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median(), inplace=True)

df_encoded = df_clean.copy()
encoders = {}

# Crear LabelEncoders exactamente igual que en el notebook
for col in df_encoded.select_dtypes(include="object").columns:
    if col != "Churn":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

df_encoded["Churn"] = df_encoded["Churn"].map({"No": 0, "Yes": 1})

feature_columns = df_encoded.drop("Churn", axis=1).columns

# Fila base para asignar valores por defecto
base_row = df_encoded.drop("Churn", axis=1).mode().iloc[0].copy()

# -------------------------------------------------
# Endpoint de predicción REAL
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Crear copia de fila base
    row = base_row.copy()

    # -------------------------
    # tenure
    # -------------------------
    if "tenure" in data and data["tenure"] not in ["", None]:
        row["tenure"] = float(data["tenure"])

    # -------------------------
    # MonthlyCharges
    # -------------------------
    if "MonthlyCharges" in data and data["MonthlyCharges"] not in ["", None]:
        row["MonthlyCharges"] = float(data["MonthlyCharges"])

    # -------------------------
    # Contract (texto o número)
    # -------------------------
    if "Contract" in data and data["Contract"] is not None:
        
        contract_value = data["Contract"]

        # Si llega un número desde el HTML
        numeric_map = {
            "0": "Month-to-month",
            "1": "One year",
            "2": "Two year",
            0: "Month-to-month",
            1: "One year",
            2: "Two year"
        }

        # Convertir si es número
        if contract_value in numeric_map:
            contract_value = numeric_map[contract_value]

        # Usar el LabelEncoder correspondiente
        contract_encoder = encoders["Contract"]

        try:
            row["Contract"] = contract_encoder.transform([contract_value])[0]
        except Exception as e:
            return jsonify({
                "error": f"El valor de Contract no es válido: {contract_value}",
                "detalles": str(e)
            }), 400

    # Convertir en DataFrame
    X_input = pd.DataFrame([row], columns=feature_columns)

    # Escalar
    X_scaled = scaler.transform(X_input)

    # Predecir
    prob = float(model.predict_proba(X_scaled)[0][1])
    pred = int(model.predict(X_scaled)[0])

    return jsonify({
        "prediction": pred,
        "probability": prob
    })


# -------------------------------------------------
# Iniciar servidor
# -------------------------------------------------
if __name__ == "__main__":
    print("✅ API de churn corriendo en http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)
