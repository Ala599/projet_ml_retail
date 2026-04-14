
from flask import Flask, request, render_template_string
import sys
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / 'src'))


DATA_TRAIN_TEST = BASE_DIR / 'data' / 'train_test'

from utils import load_model
from predict import predict_client          

app = Flask(__name__)


try:
    prep   = joblib.load(BASE_DIR / 'models' / 'preprocessor.pkl')
    kmeans = load_model('kmeans_model.pkl')
    clf    = load_model('churn_classifier.pkl')
    reg    = load_model('ltv_regressor.pkl')
    print(" Modèles chargés avec succès")
except Exception as e:
    print(f" Erreur chargement : {e}")
    exit()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>ML Retail — Prédiction Client</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 820px;
            margin: 40px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .form-group {
            margin: 15px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        label  { font-weight: bold; color: #555; flex: 1; }
        input  {
            flex: 2; padding: 10px;
            border: 2px solid #ddd; border-radius: 5px; font-size: 16px;
        }
        button {
            background: #667eea; color: white;
            padding: 15px 40px; border: none; border-radius: 8px;
            cursor: pointer; font-size: 18px; margin: 20px auto 0; display: block;
            transition: transform 0.2s;
        }
        button:hover { transform: scale(1.05); background: #5568d3; }
        .results {
            margin-top: 30px; padding: 25px; border-radius: 10px;
            animation: fadeIn 0.5s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .card {
            margin: 15px 0; padding: 20px; border-radius: 8px;
            background: #f8f9fa;
        }
        .clustering    { border-left: 5px solid #9b59b6; }
        .classification{ border-left: 5px solid #e74c3c; }
        .regression    { border-left: 5px solid #27ae60; }
        h3  { margin-top: 0; color: #333; }
        .value  { font-size: 28px; font-weight: bold; }
        .fidele { color: #27ae60; }
        .parti  { color: #e74c3c; }
    </style>
</head>
<body>
<div class="container">
    <h1>🛍️ Analyse Client</h1>

    <!-- Formulaire -->
    <form action="/predict" method="post">
        {% for feature in features %}
        <div class="form-group">
            <label>{{ feature }} :</label>
            <input type="number" step="any" name="{{ feature }}"
                   value="{{ defaults.get(feature, 0) }}">
        </div>
        {% endfor %}
        <button type="submit">Analyser</button>
    </form>

    <!-- Résultats -->
    {% if result %}
    <div class="results">
        <div class="card clustering">
            <h3>🔵 Clustering</h3>
            <div class="value">Groupe {{ result.cluster }}</div>
        </div>
        <div class="card classification">
            <h3>🎯 Prédiction Churn</h3>
            <div class="value {{ 'fidele' if 'FIDÈLE' in result.status else 'parti' }}">
                {{ result.status }}
            </div>
            <p>Probabilité de churn : <strong>{{ result.proba }} %</strong></p>
        </div>
        <div class="card regression">
            <h3>💰 LTV Estimée</h3>
            <div class="value fidele">{{ result.ltv }} €</div>
        </div>
    </div>
    {% endif %}
</div>
</body>
</html>
"""


ESSENTIAL = ['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg', 'Age', 'Country']
DEFAULTS  = {'Recency': 50, 'Frequency': 5, 'MonetaryTotal': 500,
             'MonetaryAvg': 100, 'Age': 35, 'Country': 0}


@app.route('/')
def home():
    return render_template_string(
        HTML_TEMPLATE,         
        features=ESSENTIAL,
        defaults=DEFAULTS,
        result=None
    )


@app.route('/predict', methods=['POST'])
def predict():
   
    user_input = {f: float(request.form[f]) for f in ESSENTIAL}

    
    result = predict_client(user_input)

    
    result['ltv']   = f"{result['ltv']:,.0f}"
    result['proba'] = f"{result['proba']:.1f}"

    return render_template_string(
        HTML_TEMPLATE,          
        features=ESSENTIAL,
        defaults=user_input,
        result=result
    )


if __name__ == '__main__':
    print(" Lancement sur http://127.0.0.1:5000")
    print("App prête : Clustering + Classification + Régression")
    app.run(debug=True, port=5000)