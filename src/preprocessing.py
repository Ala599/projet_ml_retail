

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION DES CHEMINS ───────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
DATA_RAW        = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED  = BASE_DIR / 'data' / 'processed'
DATA_TRAIN_TEST = BASE_DIR / 'data' / 'train_test'
MODELS_DIR      = BASE_DIR / 'models'
REPORTS_DIR     = BASE_DIR / 'reports'

for d in [DATA_PROCESSED, DATA_TRAIN_TEST, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print(" PRÉPROCESSING COMPLÈT — PROJET ML RETAIL")
print("=" * 75)

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 0 : CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
try:
    df = pd.read_excel(DATA_RAW / 'retail_customers_COMPLETE_CATEGORICAL.xlsx')
except FileNotFoundError:
    fichiers = list(DATA_RAW.glob('*.xlsx')) + list(DATA_RAW.glob('*.csv'))
    if not fichiers:
        raise FileNotFoundError("Aucun fichier trouvé dans data/raw/")
    df = pd.read_excel(fichiers[0]) if fichiers[0].suffix == '.xlsx' else pd.read_csv(fichiers[0])

print(f"\n Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"   Colonnes : {list(df.columns)}")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 : CONVERSION FORCÉE DES TYPES (avant tout traitement)
# ═══════════════════════════════════════════════════════════════════════════════
# Certaines colonnes numériques contiennent des dates mal formatées sous forme de texte.
# On force la conversion en numérique AVANT toute opération mathématique.

print("\n Conversion forcée des types numériques...")

# Colonnes qui DOIVENT être numériques selon le cahier des charges
numeric_columns = [
    'Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg', 'MonetaryStd',
    'MonetaryMin', 'MonetaryMax', 'TotalQuantity', 'AvgQuantityPerTransaction',
    'MinQuantity', 'MaxQuantity', 'CustomerTenureDays', 'FirstPurchaseDaysAgo',
    'PreferredDayOfWeek', 'PreferredHour', 'PreferredMonth',
    'WeekendPurchaseRatio', 'AvgDaysBetweenPurchases', 'UniqueProducts',
    'UniqueDescriptions', 'AvgProductsPerTransaction', 'UniqueCountries',
    'NegativeQuantityCount', 'ZeroPriceCount', 'CancelledTransactions',
    'ReturnRatio', 'TotalTransactions', 'UniqueInvoices', 'AvgLinesPerInvoice',
    'Age', 'SupportTicketsCount', 'SatisfactionScore'
]

for col in numeric_columns:
    if col in df.columns:
        original_dtype = df[col].dtype
        # Si la colonne est de type object (texte), on essaie de convertir
        if df[col].dtype == 'object':
            # D'abord, on remplace les valeurs qui ressemblent à des dates par NaN
            # car ce sont des erreurs de formatage
            sample_vals = df[col].dropna().astype(str)
            date_mask = sample_vals.str.contains(r'\d{2,4}[-/]\d{1,2}[-/]\d{2,4}', regex=True, na=False)
            if date_mask.any():
                print(f"   ⚠️  {col} contient {date_mask.sum()} valeurs date-like → converties en NaN")
                df[col] = df[col].replace(
                    to_replace=sample_vals[date_mask].tolist(),
                    value=np.nan
                )
            # Conversion en numérique
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"   {col} : {original_dtype} → {df[col].dtype} ({df[col].isna().sum()} NaN créés)")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 : PARSING DE RegistrationDate
# ═══════════════════════════════════════════════════════════════════════════════
if 'RegistrationDate' in df.columns:
    print("\n Parsing de RegistrationDate...")
    df['RegistrationDate'] = pd.to_datetime(
        df['RegistrationDate'],
        dayfirst=True,
        errors='coerce'
    )
    df['RegYear']     = df['RegistrationDate'].dt.year
    df['RegMonth']    = df['RegistrationDate'].dt.month
    df['RegDay']      = df['RegistrationDate'].dt.day
    df['RegWeekday']  = df['RegistrationDate'].dt.weekday
    df['RegQuarter']  = df['RegistrationDate'].dt.quarter
    df = df.drop(columns=['RegistrationDate'])
    print("     Extraction : RegYear, RegMonth, RegDay, RegWeekday, RegQuarter")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 : FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n Feature Engineering...")

# On utilise les noms exacts du dataset
recency_col = 'Recency' if 'Recency' in df.columns else None
freq_col = 'Frequency' if 'Frequency' in df.columns else None
monetary_col = 'MonetaryTotal' if 'MonetaryTotal' in df.columns else None
tenure_col = 'CustomerTenureDays' if 'CustomerTenureDays' in df.columns else None

if all([recency_col, monetary_col]):
    df['MonetaryPerDay'] = df[monetary_col] / (df[recency_col] + 1)
    print(f"     MonetaryPerDay = {monetary_col} / ({recency_col} + 1)")

if all([monetary_col, freq_col]):
    df['AvgBasketValue'] = df[monetary_col] / df[freq_col]
    print(f"     AvgBasketValue = {monetary_col} / {freq_col}")

if all([recency_col, tenure_col]):
    df['TenureRatio'] = df[recency_col] / (df[tenure_col] + 1)
    print(f"     TenureRatio = {recency_col} / ({tenure_col} + 1)")

if all(['UniqueProducts' in df.columns, freq_col]):
    df['ProdDivPerTrans'] = df['UniqueProducts'] / df[freq_col]
    print(f"     ProdDivPerTrans = UniqueProducts / {freq_col}")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 : SUPPRESSION DES FEATURES À RISQUE
# ═══════════════════════════════════════════════════════════════════════════════
leakage_features = [
    'ChurnRiskCategory',
    'ChurnRisk',
    'CustomerType',
    'AccountStatus',
    'SatisfactionScore',
    'LoyaltyLevel',
    'AgeCategory',
    'WeekendPreference',
    'WeekendPref',
    'SpendingCategory',
    'CustomerID',
    'LastLoginIP',
    'NewsletterSubscribed',
]

cols_to_drop = [f for f in leakage_features if f in df.columns]
print(f"\n  Features supprimées ({len(cols_to_drop)}):")
for f in cols_to_drop:
    print(f"   - {f}")
df = df.drop(columns=cols_to_drop)

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 : NETTOYAGE DES VALEURS ABERRANTES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n Nettoyage des valeurs aberrantes...")

for col in ['SupportTicketsCount']:
    if col in df.columns:
        mask = df[col].isin([-1, 999])
        n = mask.sum()
        if n > 0:
            df.loc[mask, col] = np.nan
            print(f"   {col} : {n} valeurs aberrantes (-1 / 999) → NaN")

for col in ['SatisfactionScore']:
    if col in df.columns:
        mask = (df[col] < 1) | (df[col] > 5)
        n = mask.sum()
        if n > 0:
            df.loc[mask, col] = np.nan
            print(f"   {col} : {n} valeurs hors [1-5] → NaN")

if 'Age' in df.columns:
    mask = (df['Age'] < 18) | (df['Age'] > 100)
    n = mask.sum()
    if n > 0:
        df.loc[mask, 'Age'] = np.nan
        print(f"   Age : {n} valeurs hors [18-100] → NaN")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 : SÉPARATION X / y
# ═══════════════════════════════════════════════════════════════════════════════
if 'Churn' not in df.columns:
    raise KeyError("La colonne cible 'Churn' est introuvable dans le dataset.")

y = df['Churn'].copy()
X = df.drop('Churn', axis=1).copy()

print(f"\n Features conservées : {X.shape[1]} colonnes")
print(f"   {list(X.columns)}")

churn_rate = y.mean() * 100
print(f"\n Taux de churn : {churn_rate:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 7 : IDENTIFICATION DES TYPES
# ═══════════════════════════════════════════════════════════════════════════════
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n Répartition : {len(num_cols)} numériques, {len(cat_cols)} catégorielles")
if cat_cols:
    print(f"   Catégorielles : {cat_cols}")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 8 : SPLIT TRAIN / TEST
# ═══════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Split : Train={X_train.shape[0]} | Test={X_test.shape[0]}")

# Sauvegarde brute
X_train.to_csv(DATA_PROCESSED / 'X_train_raw.csv', index=False)
X_test.to_csv(DATA_PROCESSED / 'X_test_raw.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 9 : IMPUTATION NUMÉRIQUE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n Imputation numérique (médiane)...")
imputer_num = SimpleImputer(strategy='median')
X_train[num_cols] = imputer_num.fit_transform(X_train[num_cols])
X_test[num_cols]  = imputer_num.transform(X_test[num_cols])

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 10 : VIF
# ═══════════════════════════════════════════════════════════════════════════════
print("\n Analyse VIF...")
try:
    vif_data = pd.DataFrame()
    vif_data["Feature"] = num_cols
    vif_data["VIF"] = [variance_inflation_factor(X_train[num_cols].values, i)
                       for i in range(len(num_cols))]
    vif_data = vif_data.sort_values('VIF', ascending=False)
    print(vif_data.head(10).to_string(index=False))
    high_vif = vif_data[vif_data['VIF'] > 10]['Feature'].tolist()
    if high_vif:
        print(f"\n  VIF > 10 : {high_vif}")
    else:
        print("\n VIF < 10 partout")
except Exception as e:
    print(f"   VIF non calculable : {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 11 : IMPUTATION + ENCODING CATÉGORIEL
# ═══════════════════════════════════════════════════════════════════════════════
label_encoders = {}
imputer_cat = None

if cat_cols:
    print(f"\n Encodage ({len(cat_cols)} variables)...")
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X_train[cat_cols] = imputer_cat.fit_transform(X_train[cat_cols])
    X_test[cat_cols]  = imputer_cat.transform(X_test[cat_cols])

    for col in cat_cols:
        le = LabelEncoder()
        all_values = pd.concat([X_train[col], X_test[col]]).astype(str).unique()
        le.fit(all_values)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col]  = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
        print(f"   {col} : {len(le.classes_)} catégories")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 12 : SCALING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n Standardisation...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 13 : ACP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n ACP (95% variance)...")
pca_full = PCA(random_state=42)
pca_full.fit(X_train_s)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_comp_95 = np.argmax(cumsum >= 0.95) + 1

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca  = pca.transform(X_test_s)

print(f"   {X_train_s.shape[1]} → {X_train_pca.shape[1]} composantes")
print(f"   Variance : {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 14 : SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n Sauvegarde...")

pd.DataFrame(X_train_s, columns=X.columns).to_csv(DATA_TRAIN_TEST / 'X_train.csv', index=False)
pd.DataFrame(X_test_s,  columns=X.columns).to_csv(DATA_TRAIN_TEST / 'X_test.csv',  index=False)
pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])]).to_csv(
    DATA_TRAIN_TEST / 'X_train_pca.csv', index=False)
pd.DataFrame(X_test_pca,  columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])]).to_csv(
    DATA_TRAIN_TEST / 'X_test_pca.csv',  index=False)

y_train.to_csv(DATA_TRAIN_TEST / 'y_train.csv', index=False)
y_test.to_csv(DATA_TRAIN_TEST / 'y_test.csv',   index=False)

joblib.dump({
    'columns': list(X.columns),
    'scaler': scaler,
    'imputer_num': imputer_num,
    'imputer_cat': imputer_cat,
    'label_encoders': label_encoders,
    'pca': pca,
    'num_cols': num_cols,
    'cat_cols': cat_cols,
}, MODELS_DIR / 'preprocessor.pkl')

print("\n" + "=" * 75)
print(" PRÉPROCESSING TERMINÉ")
print("=" * 75)
print(f"   Features : {len(X.columns)} | ACP : {X_train_pca.shape[1]} composantes")
print(f"   Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")
print(f"   Churn : {y_train.mean()*100:.1f}%")
print("=" * 75)

# Diagnostic
print("\n TOP 10 corrélations avec Churn :")
X_diag = pd.DataFrame(X_train_s, columns=X.columns)
corr = X_diag.corrwith(pd.Series(y_train.values)).abs().sort_values(ascending=False)
print(corr.head(10).to_string())