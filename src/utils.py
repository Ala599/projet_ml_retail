
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'


def save_model(model, filename):
   
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    print(f" Modèle sauvegardé : {path}")


def load_model(filename):
    
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Modèle non trouvé : {path}")
    return joblib.load(path)


def calculate_rfm_scores(df):
   
    df = df.copy()

    # Recency : plus petit = mieux (score inversé)
    df['R_Score'] = pd.qcut(
        df['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop'
    ).astype(int)  

    # Frequency : plus grand = mieux
    df['F_Score'] = pd.qcut(
        df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)  

    # Monetary : plus grand = mieux
    df['M_Score'] = pd.qcut(
        df['MonetaryTotal'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop'
    ).astype(int)  

    # Score RFM combiné (chaîne de caractères)
    df['RFM_Score'] = (
        df['R_Score'].astype(str)
        + df['F_Score'].astype(str)
        + df['M_Score'].astype(str)
    )

    return df


def get_segment_label(row):
   
    r = int(row['R_Score'])
    f = int(row['F_Score'])

    if r >= 4 and f >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Fidèles'
    elif r >= 4 and f <= 2:
        return 'Nouveaux'
    elif r <= 2 and f >= 3:
        return 'À Risque'
    else:
        return 'Dormants'


def plot_feature_importance(model, feature_names, title, filename):
   
    if not hasattr(model, 'feature_importances_'):
        print("  Ce modèle ne possède pas feature_importances_")
        return

    importances = pd.DataFrame({
        'feature':    feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importances, y='feature', x='importance',
                palette='viridis', ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    REPORTS_DIR.mkdir(exist_ok=True)
    fig.savefig(REPORTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close(fig) 
    print(f" Graphique sauvegardé : {filename}")


def generate_report(metrics, filename='report.json'):
    """Génère un rapport JSON"""
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f" Rapport sauvegardé : {path}")