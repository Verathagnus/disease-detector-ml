import pandas as pd
import numpy as np
import joblib
import os
import json
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

train_path = './dataset/train.csv'
valid_path = './dataset/valid.csv'

if not (os.path.exists(train_path) and os.path.exists(valid_path)):
    raise FileNotFoundError(
        "Expected 'dataset/train.csv' and 'dataset/valid.csv'. "
        "Please prepare data from the Streamlit app first."
    )

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)

data = pd.concat([train_df, valid_df], ignore_index=True)
print(f"Loaded {len(train_df)} training samples and {len(valid_df)} validation samples")

target_col = 'disease_diagnosis'
if target_col not in data.columns:
    raise ValueError(f"Target column '{target_col}' missing - cannot proceed")

# Check class balance - rare diseases like Pellagra need enough samples
disease_counts = data[target_col].value_counts()
if disease_counts.min() < 10:
    print(f"Warning: Some disease classes have very few samples (min: {disease_counts.min()})")

if data[target_col].isna().any():
    print(f"Warning: {data[target_col].isna().sum()} missing target values found")

num_features = data.select_dtypes(include=[np.number]).columns
cat_features = data.select_dtypes(include=['object']).columns

# Binary symptom flags and symptom counts aren't truly continuous
# Converting them prevents models from treating them as ordinal
fake_nums = []
for col in num_features:
    if col != target_col and data[col].nunique() <= 10:
        fake_nums.append(col)
        print(f"Converting {col} to categorical ({data[col].nunique()} unique values)")

for col in fake_nums:
    data[col] = data[col].astype(str)

num_features = data.select_dtypes(include=[np.number]).columns.tolist()
cat_features = data.select_dtypes(include=['object']).columns.tolist()
num_features = [c for c in num_features if c != target_col]
cat_features = [c for c in cat_features if c != target_col]

# Fill missing lab values with mean, categorical with 'Unknown'
data[num_features] = data[num_features].fillna(data[num_features].mean())
data[cat_features] = data[cat_features].fillna('Unknown')

features = data.drop(target_col, axis=1)
disease_labels = data[target_col]

# Convert disease names to numeric labels
le = LabelEncoder()
y = le.fit_transform(disease_labels)

# One-hot encode categoricals
X = pd.get_dummies(features, columns=cat_features, drop_first=True)

os.makedirs('model', exist_ok=True)
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(X.columns.tolist(), 'model/feature_columns.pkl')

# Use provided train/validation split (no further splitting)
n_train = len(train_df)
X_train = X.iloc[:n_train]
y_train = y[:n_train]
X_val = X.iloc[n_train:]
y_val = y[n_train:]

# Scale features - MultinomialNB needs non-negative values
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
joblib.dump(scaler, 'model/scaler.pkl')

# Best hyperparameters from GridSearchCV tuning
tuned_params = {
    "Logistic Regression": {'C': 200, 'max_iter': 1000, 'solver': 'lbfgs'},
    "Decision Tree": {'max_depth': 15, 'min_samples_leaf': 2, 'min_samples_split': 5},
    "KNN": {'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'distance'},
    "Naive Bayes (Gaussian)": {'var_smoothing': 1e-08},
    "Naive Bayes (Multinomial)": {'alpha': 2.0},
    "Random Forest": {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200},
    "XGBoost": {'learning_rate': 0.2, 'max_depth': 2, 'n_estimators': 50, 'subsample': 0.8}
}

# Override with latest tuning results if available
if os.path.exists('dataset/best_hyperparameters.json'):
    with open('dataset/best_hyperparameters.json', 'r') as f:
        latest = json.load(f)
        tuned_params.update(latest)
    print("Using hyperparameters from latest GridSearchCV run")

models = {
    "Logistic Regression": LogisticRegression(**tuned_params["Logistic Regression"]),
    "Decision Tree": DecisionTreeClassifier(**tuned_params["Decision Tree"]),
    "KNN": KNeighborsClassifier(**tuned_params["KNN"]),
    "Naive Bayes (Gaussian)": GaussianNB(**tuned_params["Naive Bayes (Gaussian)"]),
    "Naive Bayes (Multinomial)": MultinomialNB(**tuned_params["Naive Bayes (Multinomial)"]),
    "Random Forest": RandomForestClassifier(**tuned_params["Random Forest"], random_state=42),
    "XGBoost": XGBClassifier(**tuned_params["XGBoost"], eval_metric='logloss', random_state=42)
}

output = []

for name, model in tqdm(models.items(), desc="Training models"):
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Save model with cleaned filename
    fname = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    joblib.dump(model, f"./model/{fname}.pkl")
    
    preds = model.predict(X_val)
    
    auc = 0.5
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_val)
            auc = roc_auc_score(y_val, probs, multi_class='ovr')
        except ValueError:
            pass

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, average='weighted', zero_division=0)
    rec = recall_score(y_val, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_val, preds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_val, preds)

    output.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "MCC Score": round(mcc, 4)
    })

df_results = pd.DataFrame(output)
df_results = df_results[["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC Score"]]
df_results.to_csv("model_comparison_table.csv", index=False)
print(f"\nSaved results to model_comparison_table.csv")
best_model = df_results.loc[df_results['Accuracy'].idxmax(), 'ML Model Name']
print(f"Best model: {best_model} (Accuracy: {df_results['Accuracy'].max():.4f})")