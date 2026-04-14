
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / 'data' / 'raw'
REPORTS = BASE_DIR / 'reports'
REPORTS.mkdir(parents=True, exist_ok=True)

print("="*70)
print(" EXPLORATION - retail_customers_COMPLETE_CATEGORICAL")
print("="*70)


df = pd.read_excel(DATA_RAW / 'retail_customers_COMPLETE_CATEGORICAL.xlsx')

print(f"\n Dataset chargée: {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\n Colonnes ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\n Types de données:")
print(df.dtypes)

print(f"\n Valeurs manquantes:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "   Aucune valeur manquante")

print(f"\n Aperçu des données:")
print(df.head())

print(f"\n Statistiques descriptives (numériques):")
print(df.describe())

# Analyse des variables catégorielles (puisque CATEGORICAL dans le nom)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    print(f"\n Variables catégorielles trouvées ({len(cat_cols)}):")
    for col in cat_cols:
        print(f"\n   {col}:")
        print(df[col].value_counts().head())

# Visualisation de la distribution de Churn si présente
if 'Churn' in df.columns:
    plt.figure(figsize=(8, 5))
    df['Churn'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Distribution du Churn', fontweight='bold')
    plt.xlabel('Churn (0=Fidèle, 1=Parti)')
    plt.ylabel('Nombre')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS / 'churn_distribution.png', dpi=300)
    print(f"\n Graphique Churn sauvegardé")

# Matrice de corrélation des variables numériques
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) > 1:
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Matrice de Corrélation', fontweight='bold')
    plt.tight_layout()
    plt.savefig(REPORTS / 'correlation_matrix.png', dpi=300)
    print(f" Matrice de corrélation sauvegardée")

print("\n" + "="*70)
print(" Exploration terminée!")
print(f" Graphiques dans: {REPORTS}")
print("="*70)