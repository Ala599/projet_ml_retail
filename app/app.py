

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
app = Flask(__name__)
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'

# ─── CHARGEMENT DES MODÈLES ──────────────────────────────────────────────────
try:
    preprocessor = joblib.load(MODELS_DIR / 'preprocessor.pkl')
    clf = joblib.load(MODELS_DIR / 'churn_classifier.pkl')
    print("✅ Modèles chargés avec succès")
except FileNotFoundError as e:
    print(f"❌ Erreur chargement modèle : {e}")
    print("   Assurez-vous d'avoir exécuté preprocessing.py et train_model.py d'abord.")
    raise

# Récupération des infos du preprocessor
feature_names = preprocessor['columns']
num_cols = preprocessor['num_cols']
cat_cols = preprocessor['cat_cols']
scaler = preprocessor['scaler']
imputer_num = preprocessor['imputer_num']
imputer_cat = preprocessor['imputer_cat']
label_encoders = preprocessor['label_encoders']

# ─── FONCTION DE PRÉTRAITEMENT POUR UNE PRÉDICTION ──────────────────────────
def preprocess_input(data_dict):
    """
    Applique exactement le même prétraitement que pendant l'entraînement :
    imputation → encodage → scaling
    """
    df = pd.DataFrame([data_dict])

    # Vérifier que toutes les features sont présentes
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan

    # Réordonner selon l'entraînement
    df = df[feature_names]

    # 1. Imputation numérique
    df[num_cols] = imputer_num.transform(df[num_cols])

    # 2. Imputation + encodage catégoriel
    if cat_cols and imputer_cat is not None:
        df[cat_cols] = imputer_cat.transform(df[cat_cols])
        for col in cat_cols:
            le = label_encoders[col]
            val = str(df[col].iloc[0])
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                # Valeur inconnue → on met la classe la plus fréquente (index 0)
                df[col] = 0

    # 3. Scaling
    X_scaled = scaler.transform(df)

    return X_scaled

# ─── PAGE D'ACCUEIL ──────────────────────────────────────────────────────────
HTML_FORM = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Retail — Prédiction Churn</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 600px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { color: #333; margin-bottom: 10px; font-size: 28px; }
        .subtitle { color: #666; margin-bottom: 30px; font-size: 14px; }
        .form-group { margin-bottom: 15px; }
        label {
            display: block;
            margin-bottom: 5px;
            color: #444;
            font-weight: 600;
            font-size: 13px;
        }
        input, select {
            width: 100%;
            padding: 10px 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            display: none;
        }
        .result.risk-high {
            background: #ffebee;
            color: #c62828;
            border: 2px solid #ef9a9a;
        }
        .result.risk-low {
            background: #e8f5e9;
            color: #2e7d32;
            border: 2px solid #a5d6a7;
        }
        .probability {
            font-size: 36px;
            font-weight: 700;
            margin: 10px 0;
        }
        .footer {
            margin-top: 20px;
            text-align: center;
            color: #999;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛍️ ML Retail</h1>
        <p class="subtitle">Prédiction du risque de départ client (Churn)</p>

        <form id="predictForm">
            <div class="grid-2">
                <div class="form-group">
                    <label>Recency (jours depuis dernier achat)</label>
                    <input type="number" name="Recency" value="30" min="0" max="400">
                </div>
                <div class="form-group">
                    <label>Frequency (nb commandes)</label>
                    <input type="number" name="Frequency" value="5" min="1" max="50">
                </div>
            </div>

            <div class="grid-2">
                <div class="form-group">
                    <label>MonetaryTotal (£)</label>
                    <input type="number" name="MonetaryTotal" value="500" step="0.01">
                </div>
                <div class="form-group">
                    <label>MonetaryAvg (£)</label>
                    <input type="number" name="MonetaryAvg" value="100" step="0.01">
                </div>
            </div>

            <div class="grid-2">
                <div class="form-group">
                    <label>CustomerTenure (jours)</label>
                    <input type="number" name="CustomerTenure" value="365" min="0">
                </div>
                <div class="form-group">
                    <label>Age</label>
                    <input type="number" name="Age" value="35" min="18" max="100">
                </div>
            </div>

            <div class="grid-2">
                <div class="form-group">
                    <label>TotalQuantity</label>
                    <input type="number" name="TotalQuantity" value="50">
                </div>
                <div class="form-group">
                    <label>UniqueProducts</label>
                    <input type="number" name="UniqueProducts" value="10" min="1">
                </div>
            </div>

            <div class="grid-2">
                <div class="form-group">
                    <label>WeekendRatio</label>
                    <input type="number" name="WeekendRatio" value="0.2" min="0" max="1" step="0.01">
                </div>
                <div class="form-group">
                    <label>Satisfaction (1-5)</label>
                    <input type="number" name="Satisfaction" value="4" min="1" max="5" step="0.1">
                </div>
            </div>

            <button type="submit">🔮 Prédire le risque de churn</button>
        </form>

        <div id="result" class="result">
            <div id="riskText"></div>
            <div class="probability" id="probValue"></div>
            <div id="interpretation"></div>
        </div>

        <div class="footer">
            Projet ML Retail — Atelier Machine Learning GI2
        </div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = parseFloat(value);
            });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();

                const resultDiv = document.getElementById('result');
                const prob = (result.churn_probability * 100).toFixed(1);

                resultDiv.style.display = 'block';
                if (result.risk === 'Élevé') {
                    resultDiv.className = 'result risk-high';
                    document.getElementById('riskText').textContent = '⚠️ Risque ÉLEVÉ de départ';
                    document.getElementById('interpretation').textContent = 
                        'Action recommandée : campagne de rétention urgente';
                } else {
                    resultDiv.className = 'result risk-low';
                    document.getElementById('riskText').textContent = '✅ Risque FAIBLE de départ';
                    document.getElementById('interpretation').textContent = 
                        'Client fidèle — maintenir la relation';
                }
                document.getElementById('probValue').textContent = prob + '%';

            } catch (err) {
                alert('Erreur : ' + err.message);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Page d'accueil avec formulaire de prédiction."""
    return render_template_string(HTML_FORM)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API REST de prédiction churn.

    Exemple d'appel (JSON) :
    {
        "Recency": 30,
        "Frequency": 5,
        "MonetaryTotal": 500,
        "MonetaryAvg": 100,
        "CustomerTenure": 365,
        "Age": 35,
        "TotalQuantity": 50,
        "UniqueProducts": 10,
        "WeekendRatio": 0.2,
        "Satisfaction": 4
    }
    """
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({'error': 'Aucune donnée reçue'}), 400

        # Prétraitement
        X = preprocess_input(data)

        # Prédiction
        proba = clf.predict_proba(X)[0, 1]
        prediction = int(clf.predict(X)[0])

        # Seuil de décision (on peut ajuster, 0.5 par défaut)
        threshold = 0.5
        risk = 'Élevé' if proba >= threshold else 'Faible'

        return jsonify({
            'churn_probability': round(float(proba), 4),
            'churn_prediction': prediction,
            'risk': risk,
            'threshold': threshold,
            'model': 'RandomForestClassifier',
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de vérification de santé."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'preprocessor': preprocessor is not None,
            'classifier': clf is not None
        }
    })

@app.route('/features', methods=['GET'])
def features():
    """Retourne la liste des features attendues."""
    return jsonify({
        'features': feature_names,
        'numerical': num_cols,
        'categorical': cat_cols
    })

# ─── LANCEMENT ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("🌐 ML RETAIL — Application Flask")
    print("=" * 60)
    print("Routes disponibles :")
    print("   GET  /           → Interface web")
    print("   POST /predict    → API prédiction churn (JSON)")
    print("   GET  /health     → Vérification santé")
    print("   GET  /features   → Liste des features")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)