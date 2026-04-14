
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).parent))
from utils import load_model

BASE_DIR        = Path(__file__).parent.parent
MODELS_DIR      = BASE_DIR / 'models'
DATA_TRAIN_TEST = BASE_DIR / 'data' / 'train_test'
DATA_RAW        = BASE_DIR / 'data' / 'raw'


def preprocess_single(raw_values: dict) -> pd.DataFrame:
    
    prep     = joblib.load(MODELS_DIR / 'preprocessor.pkl')
    columns  = prep['columns']
    scaler   = prep['scaler']

    
    X_train_means = pd.read_csv(DATA_TRAIN_TEST / 'X_train.csv').mean().to_dict()

    full = {col: X_train_means.get(col, 0.0) for col in columns}
    full.update({k: v for k, v in raw_values.items() if k in columns})

    df = pd.DataFrame([full])[columns]
    df_scaled = pd.DataFrame(scaler.transform(df), columns=columns)
    return df_scaled


def predict_client(raw_values: dict) -> dict:
    
    df_scaled = preprocess_single(raw_values)

    kmeans = load_model('kmeans_model.pkl')
    clf    = load_model('churn_classifier.pkl')
    reg    = load_model('ltv_regressor.pkl')

    cluster = int(kmeans.predict(df_scaled)[0])
    churn   = int(clf.predict(df_scaled)[0])
    proba   = float(clf.predict_proba(df_scaled)[0][1])
    ltv     = float(reg.predict(df_scaled)[0])

    status = " FIDÈLE" if churn == 0 else " PARTI"

    print(f"\n1️  Cluster  : Groupe {cluster}")
    print(f"2️  Churn    : {status} ({proba * 100:.1f}%)")
    print(f"3️  LTV      : {ltv:,.0f} €")

    
    return {
        'cluster': cluster,
        'status':  status,
        'proba':   round(proba * 100, 1),
        'ltv':     round(ltv, 2),
    }


# ── Tests ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    
    print("TEST 1 : Client normal (valeurs moyennes)")
    predict_client({
        'Recency':       50,
        'Frequency':      8,
        'MonetaryTotal': 600,
        'MonetaryAvg':    75,
        'Age':            35,
    })

    print("\nTEST 2 : Client à risque (recency élevée, peu d'achats)")
   
    predict_client({
        'Recency':       300,   # Très inactif
        'Frequency':       1,   # Un seul achat
        'MonetaryTotal':  80,
        'MonetaryAvg':    80,
        'Age':            45,
    })

    print("\nTEST 3 : Client VIP (fort acheteur récent)")
    predict_client({
        'Recency':         5,
        'Frequency':      40,
        'MonetaryTotal': 8000,
        'MonetaryAvg':    200,
        'Age':             29,
    })