from flask import Flask, request, jsonify, send_from_directory
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
# Reconstruir preprocesamiento EXACTO
# -------------------------------------------------
df = pd.read_excel("Customer_Churn_Dataset.xlsx")
df_clean = df.drop("customerID", axis=1).copy()

df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median(), inplace=True)

df_encoded = df_clean.copy()
encoders = {}

# LabelEncoders EXACTOS del entrenamiento
for col in df_encoded.select_dtypes(include="object").columns:
    if col != "Churn":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

df_encoded["Churn"] = df_encoded["Churn"].map({"No": 0, "Yes": 1})

feature_columns = df_encoded.drop("Churn", axis=1).columns

# Fila base
base_row = df_encoded.drop("Churn", axis=1).mode().iloc[0].copy()

# -------------------------------------------------
# Servir HTML estático desde /static/index.html
# -------------------------------------------------
@app.get("/")
def serve_dashboard():
    return send_from_directory("static", "index.html")

@app.get("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

# -------------------------------------------------
# Endpoint de predicción REAL
# -------------------------------------------------
@app.post("/predict")
def predict():
    data = request.json

    row = base_row.copy()

    # -------------------------
    # tenure
    # -------------------------
    if "tenure" in data:
        row["tenure"] = float(data["tenure"])

    # -------------------------
    # MonthlyCharges (from HTML: "monthly")
    # -------------------------
    if "monthly" in data:
        row["MonthlyCharges"] = float(data["monthly"])

    # -------------------------
    # Contract
    # -------------------------
    if "contract" in data:

        contract_value = data["contract"]

        # Si HTML envía 0,1,2 → convertir a texto original
        numeric_map = {
            "0": "Month-to-month",
            "1": "One year",
            "2": "Two year",
            0: "Month-to-month",
            1: "One year",
            2: "Two year"
        }

        if contract_value in numeric_map:
            contract_value = numeric_map[contract_value]

        le_contract = encoders["Contract"]

        try:
            row["Contract"] = le_contract.transform([contract_value])[0]
        except:
            return jsonify({
                "error": f"Valor de Contract inválido: {contract_value}"
            }), 400

    # -------------------------
    # Crear df
    # -------------------------
    X_input = pd.DataFrame([row], columns=feature_columns)

    # Escalar
    X_scaled = scaler.transform(X_input)

    # Predecir
    prob = float(model.predict_proba(X_scaled)[0][1])
    pred = int(model.predict(X_scaled)[0])

    return jsonify({
        "prediction": pred,
        "prob": prob
    })

# -------------------------------------------------
# Iniciar servidor
# -------------------------------------------------
if __name__ == "__main__":
    print("🚀 API de churn corriendo en http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)
