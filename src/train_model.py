
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score)
import sys
import joblib

sys.path.append(str(Path(__file__).parent))
from utils import save_model, plot_feature_importance

BASE_DIR       = Path(__file__).parent.parent
DATA_TRAIN_TEST = BASE_DIR / 'data' / 'train_test'

print("=" * 70)
print(" ENTRAÎNEMENT (CORRIGÉ SANS DATA LEAKAGE)")
print("=" * 70)

# ── Chargement ───────────────────────────────────────────────────────────────
X_train = pd.read_csv(DATA_TRAIN_TEST / 'X_train.csv')
X_test  = pd.read_csv(DATA_TRAIN_TEST / 'X_test.csv')
y_train = pd.read_csv(DATA_TRAIN_TEST / 'y_train.csv').squeeze()
y_test  = pd.read_csv(DATA_TRAIN_TEST / 'y_test.csv').squeeze()

print(f" Features utilisées : {X_train.shape[1]}")
print(f"   Exemples : {list(X_train.columns[:5])}…")

# =============================================================================
# 1. CLUSTERING
# =============================================================================
print("\n" + "=" * 70)
print("1️  CLUSTERING")
print("=" * 70)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_train = kmeans.fit_predict(X_train)

print("Distribution clusters :", pd.Series(clusters_train).value_counts().to_dict())
save_model(kmeans, 'kmeans_model.pkl')

# =============================================================================
# 2. CLASSIFICATION — Churn
# =============================================================================
print("\n" + "=" * 70)
print("2️  CLASSIFICATION")
print("=" * 70)

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight='balanced',
    min_samples_leaf=10,
    random_state=42
)
clf.fit(X_train, y_train)

y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print(f"\n Accuracy : {clf.score(X_test, y_test):.3f}")
print("\nClassification Report :")
print(classification_report(y_test, y_pred, target_names=['Fidèle', 'Parti']))

test_client = X_test.iloc[0:1].copy()
pred  = clf.predict(test_client)[0]
proba = clf.predict_proba(test_client)[0][1]
print(f"\n Exemple réel du dataset :")
print(f"   Prédiction : {' PARTI' if pred == 1 else ' FIDÈLE'} ({proba * 100:.1f}%)")

save_model(clf, 'churn_classifier.pkl')
plot_feature_importance(clf, X_train.columns, 'Importance Features', 'feat_imp.png')

# =============================================================================
# 3. RÉGRESSION — LTV estimé
# =============================================================================
print("\n" + "=" * 70)
print("3️  RÉGRESSION")
print("=" * 70)


for dataset_name, dataset in [('X_train', X_train), ('X_test', X_test)]:
    missing = [c for c in ['MonetaryTotal', 'Frequency', 'Recency'] if c not in dataset.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {dataset_name} : {missing}")

ltv_train = (X_train['MonetaryTotal'] * X_train['Frequency']
             / (X_train['Recency'] + 1))
ltv_test  = (X_test['MonetaryTotal']  * X_test['Frequency']
             / (X_test['Recency'] + 1))

reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
reg.fit(X_train, ltv_train)

ltv_pred = reg.predict(X_test)
mae = mean_absolute_error(ltv_test, ltv_pred)
r2  = r2_score(ltv_test, ltv_pred)

print(f"MAE : {mae:.2f}")
print(f"R²  : {r2:.3f}")

save_model(reg, 'ltv_regressor.pkl')

print("\n" + "=" * 70)
print(" 3 MODÈLES SAUVEGARDÉS AVEC SUCCÈS")
print("=" * 70)