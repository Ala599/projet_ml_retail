
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_TRAIN_TEST = BASE_DIR / 'data' / 'train_test'
MODELS_DIR = BASE_DIR / 'models'

for d in [DATA_TRAIN_TEST, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("🧹 PRÉPROCESSING FINAL (SANS DATA LEAKAGE)")
print("=" * 70)

# ── Chargement ──────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_RAW / 'retail_customers_COMPLETE_CATEGORICAL.xlsx')

# ── Nettoyage des types (dates dans champs numériques) ───────────────────────
for col in df.columns:
    if df[col].dtype == 'object':
        sample = df[col].astype(str).iloc[0]
        if '-' in sample and len(sample) > 8:
            df[col] = 0  

# ── Suppression des features à risque de data leakage ───────────────────────
leakage_features = [
    'ChurnRiskCategory', 'RFMSegment', 'CustomerType',
    'CustomerID', 'LastLoginIP', 'RegistrationDate', 'NewsletterSubscribed',
    'AccountStatus',       
    'SatisfactionScore',   
    'SupportTicketsCount', 
    'AgeCategory',         
    'LoyaltyLevel',        
    'WeekendPreference'    
]
df = df.drop(columns=[f for f in leakage_features if f in df.columns])

# ── Séparation X / y ────────────────────────────────────────────────────────
y = df['Churn']
X = df.drop('Churn', axis=1)

print(f"Features propres : {X.shape[1]} colonnes")

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Imputation numérique ──────────────────────────────────────────────────────
imputer_num = SimpleImputer(strategy='median')
X_train[num_cols] = imputer_num.fit_transform(X_train[num_cols])
X_test[num_cols]  = imputer_num.transform(X_test[num_cols])

# ── Imputation catégorielle + encoding ──────────────────────────────────────
label_encoders = {}
if cat_cols:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X_train[cat_cols] = imputer_cat.fit_transform(X_train[cat_cols])
    X_test[cat_cols]  = imputer_cat.transform(X_test[cat_cols])

    for col in cat_cols:
        le = LabelEncoder()
        
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        
        X_test[col] = X_test[col].astype(str).map(
            lambda val: le.transform([val])[0]
            if val in le.classes_
            else le.transform([le.classes_[0]])[0]
        )
        label_encoders[col] = le

# ── Scaling (fit sur train uniquement) ──────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Sauvegarde des datasets ──────────────────────────────────────────────────
pd.DataFrame(X_train_s, columns=X.columns).to_csv(DATA_TRAIN_TEST / 'X_train.csv', index=False)
pd.DataFrame(X_test_s,  columns=X.columns).to_csv(DATA_TRAIN_TEST / 'X_test.csv',  index=False)
y_train.to_csv(DATA_TRAIN_TEST / 'y_train.csv', index=False)
y_test.to_csv(DATA_TRAIN_TEST / 'y_test.csv',   index=False)

# ── Sauvegarde du preprocessor complet ──────────────────────────────────────
joblib.dump({
    'columns':        list(X.columns),
    'scaler':         scaler,
    'imputer_num':    imputer_num,
    'imputer_cat':    imputer_cat if cat_cols else None,
    'label_encoders': label_encoders,
    'num_cols':       num_cols,
    'cat_cols':       cat_cols,
}, MODELS_DIR / 'preprocessor.pkl')

print(f" Terminé : {len(X.columns)} features")
print(f"   Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")
print(f"   Colonnes : {list(X.columns)}")