

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─── SKLEARN ─────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import PCA

# ─── IMBALANCED-LEARN (optionnel, pour comparaison) ──────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  imblearn non installé. SMOTE sera ignoré.")

# ─── VISUALISATION ───────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
DATA_TRAIN_TEST = BASE_DIR / 'data' / 'train_test'
MODELS_DIR      = BASE_DIR / 'models'
REPORTS_DIR     = BASE_DIR / 'reports'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).parent))

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════
def save_model(model, filename):
    """Sauvegarde un modèle entraîné avec joblib."""
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    print(f"💾 Modèle sauvegardé : models/{filename}")

def plot_feature_importance(model, feature_names, title, filename, top_n=15):
    """Affiche et sauvegarde l'importance des features."""
    if not hasattr(model, 'feature_importances_'):
        return
    feat_imp = pd.Series(model.feature_importances_, index=feature_names)
    feat_imp = feat_imp.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(feat_imp))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette=colors)
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Graphique sauvegardé : reports/{filename}")

# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 75)
print("🚀 ENTRAÎNEMENT DES MODÈLES — PROJET ML RETAIL")
print("=" * 75)

# On charge la version SCALÉE (X_train.csv) et la version ACP (X_train_pca.csv)
X_train    = pd.read_csv(DATA_TRAIN_TEST / 'X_train.csv')
X_test     = pd.read_csv(DATA_TRAIN_TEST / 'X_test.csv')
X_train_pca = pd.read_csv(DATA_TRAIN_TEST / 'X_train_pca.csv')
X_test_pca  = pd.read_csv(DATA_TRAIN_TEST / 'X_test_pca.csv')
y_train    = pd.read_csv(DATA_TRAIN_TEST / 'y_train.csv').squeeze()
y_test     = pd.read_csv(DATA_TRAIN_TEST / 'y_test.csv').squeeze()

feature_names = list(X_train.columns)
print(f"\n✅ Données chargées : {X_train.shape[1]} features | {X_train.shape[0]} échantillons")
print(f"   Composantes ACP    : {X_train_pca.shape[1]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CLUSTERING — K-MEANS (Segmentation clients)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("1️⃣  CLUSTERING K-MEANS — SEGMENTATION CLIENTS")
print("=" * 75)
print("Objectif : Regrouper les clients en segments homogènes pour le marketing.")

# ── Méthode du coude ─────────────────────────────────────────────────────────
print("\n📉 Recherche du K optimal (méthode du coude)...")
inertias = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_train_pca)          # On utilise l'ACP pour le clustering (plus rapide)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Nombre de clusters K')
plt.ylabel('Inertie (WCSS)')
plt.title('Méthode du Coude — Choix du K optimal')
plt.xticks(K_range)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'elbow_curve.png', dpi=150)
plt.close()
print("   📊 Courbe du coude → reports/elbow_curve.png")

# Choix du K (on peut aussi utiliser le silhouette score, mais K=4 est cohérent)
K_OPTIMAL = 4
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
clusters_train = kmeans.fit_predict(X_train_pca)
clusters_test  = kmeans.predict(X_test_pca)

# Distribution
print(f"\n📊 Distribution des {K_OPTIMAL} clusters (train) :")
dist = pd.Series(clusters_train).value_counts().sort_index()
for k, n in dist.items():
    print(f"   Groupe {k} : {n} clients ({n/len(clusters_train)*100:.1f}%)")

# Visualisation 2D via ACP
pca_viz = PCA(n_components=2, random_state=42)
X_2d = pca_viz.fit_transform(X_train)

plt.figure(figsize=(9, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters_train,
                      cmap='tab10', alpha=0.5, s=15)
plt.colorbar(scatter, label='Cluster')
plt.title(f'Visualisation des Clusters K-Means (K={K_OPTIMAL}) — PCA 2D')
plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'clusters_pca.png', dpi=150)
plt.close()
print("   📊 Clusters PCA 2D → reports/clusters_pca.png")

save_model(kmeans, 'kmeans_model.pkl')

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CLASSIFICATION — PRÉDICTION DU CHURN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("2️⃣  CLASSIFICATION — PRÉDICTION DU CHURN")
print("=" * 75)
print("Objectif : Prédire si un client va partir (1) ou rester (0).")

# ── 2A. Référence : Régression Logistique (baseline) ─────────────────────────
print("\n📌 Baseline : Régression Logistique")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
lr_proba = lr.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_proba)
print(f"   ROC-AUC (Logistic) : {lr_auc:.3f}")

# ── 2B. Random Forest avec GridSearchCV ──────────────────────────────────────
print("\n🔍 GridSearchCV — Random Forest (peut prendre 1-2 minutes)...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6, 8, 12, None],
    'min_samples_leaf': [5, 10, 20],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf_grid = RandomForestClassifier(random_state=42, n_jobs=-1)
grid = GridSearchCV(
    rf_grid,
    param_grid,
    cv=5,                       # Validation croisée 5-fold
    scoring='roc_auc',          # Métrique principale pour le churn
    n_jobs=-1,
    verbose=0
)
grid.fit(X_train, y_train)

print(f"\n✅ Meilleurs hyperparamètres :")
for param, val in grid.best_params_.items():
    print(f"   {param} = {val}")
print(f"   Meilleur ROC-AUC (CV) : {grid.best_score_:.3f}")

clf = grid.best_estimator_

# ── Prédictions ──────────────────────────────────────────────────────────────
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# ── Métriques ────────────────────────────────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n📈 Métriques finales (test set) :")
print(f"   Accuracy  : {accuracy:.3f}")
print(f"   Precision : {precision:.3f}")
print(f"   Recall    : {recall:.3f}")
print(f"   F1-Score  : {f1:.3f}")
print(f"   ROC-AUC   : {roc_auc:.3f}  ← métrique principale")

# Interprétation du ROC-AUC
if roc_auc >= 0.90:
    print("   ⚠️  ROC-AUC > 0.90 → Vérifier s'il reste du data leakage !")
elif roc_auc >= 0.75:
    print("   ✅ ROC-AUC entre 0.75-0.90 → Bon modèle réaliste.")
else:
    print("   ⚠️  ROC-AUC < 0.75 → Modèle à améliorer (feature engineering ?)")

print("\n📋 Rapport de classification :")
print(classification_report(y_test, y_pred,
                            target_names=['Fidèle (0)', 'Parti (1)']))

# ── Matrice de confusion ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fidèle', 'Parti'],
            yticklabels=['Fidèle', 'Parti'])
plt.title('Matrice de Confusion — Churn')
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'confusion_matrix.png', dpi=150)
plt.close()
print("\n📊 Matrice de confusion → reports/confusion_matrix.png")

# ── Courbe ROC ───────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC — Prédiction Churn')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'roc_curve.png', dpi=150)
plt.close()
print("📊 Courbe ROC → reports/roc_curve.png")

# ── Feature Importance ───────────────────────────────────────────────────────
plot_feature_importance(
    clf, feature_names,
    title='Top 15 Features — Importance pour la Prédiction Churn',
    filename='feature_importance_churn.png',
    top_n=15
)

save_model(clf, 'churn_classifier.pkl')

# ── 2C. Comparaison avec SMOTE (sur-échantillonnage) ─────────────────────────
if SMOTE_AVAILABLE:
    print("\n🔄 Comparaison avec SMOTE (sur-échantillonnage)...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print(f"   Distribution après SMOTE : {np.bincount(y_train_sm)}")

    clf_smote = RandomForestClassifier(
        n_estimators=200,
        max_depth=grid.best_params_.get('max_depth', 8),
        min_samples_leaf=grid.best_params_.get('min_samples_leaf', 10),
        class_weight='balanced',
        random_state=42
    )
    clf_smote.fit(X_train_sm, y_train_sm)
    y_proba_sm = clf_smote.predict_proba(X_test)[:, 1]
    roc_auc_sm = roc_auc_score(y_test, y_proba_sm)

    print(f"   ROC-AUC (SMOTE) : {roc_auc_sm:.3f}")
    print(f"   ROC-AUC (base)  : {roc_auc:.3f}")
    if roc_auc_sm > roc_auc:
        print("   ✅ SMOTE améliore les performances")
    else:
        print("   ℹ️  SMOTE n'apporte pas de gain significatif ici")

    save_model(clf_smote, 'churn_classifier_smote.pkl')

# ═══════════════════════════════════════════════════════════════════════════════
# 3. RÉGRESSION — ESTIMATION DE LA LTV (Lifetime Value)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("3️⃣  RÉGRESSION — ESTIMATION LTV (Lifetime Value)")
print("=" * 75)
print("Objectif : Estimer la valeur financière future d'un client.")
print("Formule LTV = MonetaryTotal × Frequency / (Recency + 1)")

# Construction de la cible LTV
if all(c in X_train.columns for c in ['MonetaryTotal', 'Frequency', 'Recency']):
    ltv_train = (X_train['MonetaryTotal'].values * X_train['Frequency'].values
                 / (X_train['Recency'].values + 1))
    ltv_test  = (X_test['MonetaryTotal'].values * X_test['Frequency'].values
                 / (X_test['Recency'].values + 1))
else:
    # Fallback si les colonnes ont été renommées
    print("   ⚠️  Colonnes RFM non trouvées — LTV simulée pour démonstration")
    ltv_train = np.random.lognormal(3, 1, len(X_train))
    ltv_test  = np.random.lognormal(3, 1, len(X_test))

print(f"\n   LTV train — Moyenne : {ltv_train.mean():.1f} | Médiane : {np.median(ltv_train):.1f}")
print(f"   LTV test  — Moyenne : {ltv_test.mean():.1f}  | Médiane : {np.median(ltv_test):.1f}")

# ── GridSearchCV pour la régression ──────────────────────────────────────────
print("\n🔍 GridSearchCV — Random Forest Regressor...")
param_grid_reg = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12, None],
    'min_samples_leaf': [2, 5, 10]
}

rf_reg_grid = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_reg = GridSearchCV(
    rf_reg_grid,
    param_grid_reg,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
grid_reg.fit(X_train, ltv_train)

print(f"\n✅ Meilleurs params régression : {grid_reg.best_params_}")
reg = grid_reg.best_estimator_

# ── Évaluation ───────────────────────────────────────────────────────────────
ltv_pred = reg.predict(X_test)
mae = mean_absolute_error(ltv_test, ltv_pred)
mse = mean_squared_error(ltv_test, ltv_pred)
r2  = r2_score(ltv_test, ltv_pred)

print(f"\n Métriques régression (test set) :")
print(f"   MAE : {mae:.2f}  (erreur moyenne absolue)")
print(f"   RMSE: {np.sqrt(mse):.2f} (erreur quadratique moyenne)")
print(f"   R²  : {r2:.3f}  ← proportion de variance expliquée")

if r2 >= 0.85:
    print("   ⚠️  R² > 0.85 → Vérifier si LTV n'est pas trop dérivée des features")
elif r2 >= 0.60:
    print("   ✅ R² entre 0.60-0.85 → Bonne estimation de la LTV")
else:
    print("   ⚠️  R² < 0.60 → LTV difficile à estimer avec ces features")

# ── Scatter réel vs prédit ───────────────────────────────────────────────────
plt.figure(figsize=(7, 5))
plt.scatter(ltv_test, ltv_pred, alpha=0.3, s=10, color='steelblue')
plt.plot([ltv_test.min(), ltv_test.max()],
         [ltv_test.min(), ltv_test.max()], 'r--', linewidth=1.5, label='Idéal')
plt.xlabel('LTV réelle')
plt.ylabel('LTV prédite')
plt.title(f'Régression LTV — R² = {r2:.3f} | MAE = {mae:.1f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'ltv_regression.png', dpi=150)
plt.close()
print("\n📊 Graphique LTV → reports/ltv_regression.png")

# ── Feature importance régression ────────────────────────────────────────────
plot_feature_importance(
    reg, feature_names,
    title='Top 15 Features — Importance pour la Régression LTV',
    filename='feature_importance_ltv.png',
    top_n=15
)

save_model(reg, 'ltv_regressor.pkl')

# ═══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("✅ ENTRAÎNEMENT TERMINÉ — MODÈLES ET RAPPORTS GÉNÉRÉS")
print("=" * 75)
print(f"\n   📁 Modèles sauvegardés dans models/ :")
print(f"      • kmeans_model.pkl          → Segmentation ({K_OPTIMAL} groupes)")
print(f"      • churn_classifier.pkl      → Churn (ROC-AUC = {roc_auc:.3f})")
if SMOTE_AVAILABLE:
    print(f"      • churn_classifier_smote.pkl→ Churn avec SMOTE")
print(f"      • ltv_regressor.pkl         → LTV (R² = {r2:.3f})")
print(f"\n   📁 Graphiques sauvegardés dans reports/ :")
print(f"      • elbow_curve.png")
print(f"      • clusters_pca.png")
print(f"      • confusion_matrix.png")
print(f"      • roc_curve.png")
print(f"      • feature_importance_churn.png")
print(f"      • feature_importance_ltv.png")
print(f"      • ltv_regression.png")
print("=" * 75)