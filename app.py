import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Vitamin Deficiency Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Teal theme for medical app
st.markdown("""
    <style>
    .stButton>button {
        background-color: #008080 !important;
        color: white !important;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #006666 !important;
    }
    .stFormSubmitButton>button {
        background-color: #008080 !important;
        color: white !important;
        width: 100%;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stFormSubmitButton>button:hover {
        background-color: #006666 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Vitamin Deficiency Disease Prediction")

has_data = os.path.exists('dataset/data.csv')

st.header("1. Dataset Management")
if has_data:
    st.success("Dataset already downloaded at dataset/data.csv")
else:
    st.info("Dataset not found. Click the button below to download.")

if st.button("Download Dataset"):
    if has_data:
        st.info("Dataset already exists. No download needed.")
    else:
        if not os.path.exists('download_dataset.sh'):
            st.error("download_dataset.sh script not found.")
        else:
            with st.spinner("Downloading dataset..."):
                try:
                    os.chmod('download_dataset.sh', 0o755)
                    proc = subprocess.run(
                        ['bash', 'download_dataset.sh'],
                        capture_output=True,
                        text=True,
                        cwd=os.getcwd()
                    )
                    if proc.returncode == 0:
                        if os.path.exists('dataset/data.csv'):
                            st.success("Dataset downloaded successfully!")
                            st.rerun()
                        else:
                            st.warning("Download completed but data.csv not found. Check script output.")
                    else:
                        err = proc.stderr if proc.stderr else proc.stdout
                        st.error(f"Download failed: {err}")
                except OSError as e:
                    st.error(f"Script execution error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

if has_data:
    if st.button("Delete Dataset"):
        if os.path.exists('dataset/data.csv'):
            os.remove('dataset/data.csv')
            st.success("Dataset deleted successfully!")
            st.rerun()
        else:
            st.error("Dataset not found.")

st.subheader("Dataset Preparation")
has_train = os.path.exists('dataset/train.csv') and os.path.exists('dataset/valid.csv')
has_test = os.path.exists('dataset/test.csv')

if has_train and has_test:
    st.success("Train, Validation, and Test splits already exist.")
    train_df = pd.read_csv('dataset/train.csv')
    valid_df = pd.read_csv('dataset/valid.csv')
    test_df = pd.read_csv('dataset/test.csv')
    st.info(f"Train: {len(train_df)} rows | Valid: {len(valid_df)} rows | Test: {len(test_df)} rows")
else:
    st.info("Click the button to prepare dataset (remove duplicates and create 8:1:1 split).")

if st.button("Prepare Dataset"):
    if not has_data:
        st.error("Please download the dataset first.")
    else:
        with st.spinner("Preparing dataset..."):
            try:
                raw = pd.read_csv('dataset/data.csv')
                orig_len = len(raw)
                
                dupes = raw.duplicated().sum()
                if dupes > 0:
                    raw = raw.drop_duplicates()
                    st.info(f"Removed {dupes} duplicate rows.")
                
                target = 'disease_diagnosis'
                if target not in raw.columns:
                    st.error(f"Target column '{target}' not found in dataset.")
                else:
                    # Stratified 8:1:1 split preserves disease class ratios
                    train_df, temp_df = train_test_split(
                        raw, test_size=0.2, random_state=42, stratify=raw[target]
                    )
                    valid_df, test_df = train_test_split(
                        temp_df, test_size=0.5, random_state=42, stratify=temp_df[target]
                    )
                    
                    os.makedirs('dataset', exist_ok=True)
                    train_df.to_csv('dataset/train.csv', index=False)
                    valid_df.to_csv('dataset/valid.csv', index=False)
                    test_df.to_csv('dataset/test.csv', index=False)
                    
                    st.success("Dataset prepared successfully!")
                    st.write(f"**Original:** {orig_len} rows")
                    if dupes > 0:
                        st.write(f"**After dedup:** {len(raw)} rows")
                    st.write(f"**Train:** {len(train_df)} rows (~80%)")
                    st.write(f"**Valid:** {len(valid_df)} rows (~10%)")
                    st.write(f"**Test:** {len(test_df)} rows (~10%)")
                    
                    st.write("**Class Distribution:**")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write("Train:")
                        st.dataframe(train_df[target].value_counts().sort_index(), width='stretch')
                    with c2:
                        st.write("Valid:")
                        st.dataframe(valid_df[target].value_counts().sort_index(), width='stretch')
                    with c3:
                        st.write("Test:")
                        st.dataframe(test_df[target].value_counts().sort_index(), width='stretch')
                    
                    st.rerun()
            except pd.errors.EmptyDataError:
                st.error("Dataset file is empty or corrupted.")
            except Exception as e:
                st.error(f"Preparation failed: {e}")

if has_train or has_test:
    if st.button("Delete Data Files"):
        try:
            deleted = []
            if os.path.exists('dataset/train.csv'):
                os.remove('dataset/train.csv')
                deleted.append('train.csv')
            if os.path.exists('dataset/valid.csv'):
                os.remove('dataset/valid.csv')
                deleted.append('valid.csv')
            if os.path.exists('dataset/test.csv'):
                os.remove('dataset/test.csv')
                deleted.append('test.csv')
            
            if deleted:
                st.success(f"Deleted: {', '.join(deleted)}")
                st.rerun()
            else:
                st.info("No data files to delete.")
        except PermissionError:
            st.error("Permission denied. File may be in use.")
        except Exception as e:
            st.error(f"Delete failed: {e}")

st.header("2. Model Training")
has_models = os.path.exists('model/feature_columns.pkl')

def run_training():
    if not has_data:
        st.error("Please download the dataset first.")
        return
    
    log_box = st.empty()
    log_lines = []

    with st.spinner("Training models... This may take a few minutes."):
        try:
            proc = subprocess.Popen(
                [sys.executable, 'train_models.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd()
            )

            for line in proc.stdout:
                log_lines.append(line.rstrip("\n"))
                # Show only the last 20 lines to keep UI responsive
                log_box.text("\n".join(log_lines[-20:]))

            proc.wait()

            if proc.returncode == 0:
                st.success("Models trained successfully!")
                st.rerun()
            else:
                st.error(f"Training failed with exit code {proc.returncode}. See logs above.")
        except FileNotFoundError:
            st.error("train_models.py not found in current directory.")
        except Exception as e:
            st.error(f"Training error: {e}")

if not has_models:
    st.warning("Models not trained yet. Please train models first.")
    if st.button("Train Models"):
        run_training()
else:
    st.success("Models are trained and ready to use.")
    if st.button("Retrain Models"):
        run_training()
    
    if st.button("Delete Models"):
        try:
            if os.path.exists('model'):
                pkl_files = [f for f in os.listdir('model') if f.endswith('.pkl')]
                if pkl_files:
                    for f in pkl_files:
                        os.remove(os.path.join('model', f))
                    st.success(f"Deleted {len(pkl_files)} model file(s).")
                    st.rerun()
                else:
                    st.info("No model files to delete.")
            else:
                st.info("Model directory does not exist.")
        except OSError as e:
            st.error(f"File system error: {e}")
        except Exception as e:
            st.error(f"Delete failed: {e}")

st.header("3. Model Comparison Table")
train_csv = 'model_comparison_table.csv'
test_csv = 'model_comparison_test.csv'

if st.button("Refresh Table"):
    st.rerun()

def evaluate_test_set():
    if not has_models:
        st.error("Please train models first.")
        return
    
    if not os.path.exists('dataset/test.csv'):
        st.error("Test dataset not found. Please prepare the dataset first.")
        return
    
    with st.spinner("Evaluating models on test data..."):
        try:
            test = pd.read_csv('dataset/test.csv')
            target = 'disease_diagnosis'
            
            if target not in test.columns:
                st.error(f"Test dataset must contain '{target}' column.")
                return
            
            # Load preprocessing artifacts
            feat_cols = joblib.load('model/feature_columns.pkl')
            scaler = joblib.load('model/scaler.pkl')
            le = joblib.load('model/label_encoder.pkl')
            
            # Preprocess test data (match training pipeline)
            y_true = le.transform(test[target])
            X = test.drop(columns=[target], errors='ignore')
            
            # Handle missing values
            nums = X.select_dtypes(include=[np.number]).columns
            cats = X.select_dtypes(include=['object']).columns
            X[nums] = X[nums].fillna(X[nums].mean())
            X[cats] = X[cats].fillna('Unknown')
            
            # One-hot encode and align
            X = pd.get_dummies(X, drop_first=True)
            X = X.reindex(columns=feat_cols, fill_value=0)
            X = scaler.transform(X)
            
            # Evaluate all models
            model_files = [f for f in os.listdir('model') 
                                if f.endswith('.pkl') and f not in ['scaler.pkl', 'feature_columns.pkl', 'label_encoder.pkl']]
            results = []
            
            for mfile in model_files:
                mname = mfile.replace('.pkl', '').replace('_', ' ').title()
                mpath = os.path.join('model', mfile)
                mdl = joblib.load(mpath)
                
                preds = mdl.predict(X)
                
                # AUC calculation
                auc = 0.0
                if hasattr(mdl, "predict_proba"):
                    try:
                        probs = mdl.predict_proba(X)
                        auc = roc_auc_score(y_true, probs, multi_class='ovr')
                    except ValueError:
                        pass
                
                results.append({
                    "ML Model Name": mname,
                    "Accuracy": round(accuracy_score(y_true, preds), 4),
                    "AUC": round(auc, 4),
                    "Precision": round(precision_score(y_true, preds, average='weighted', zero_division=0), 4),
                    "Recall": round(recall_score(y_true, preds, average='weighted', zero_division=0), 4),
                    "F1 Score": round(f1_score(y_true, preds, average='weighted', zero_division=0), 4),
                    "MCC Score": round(matthews_corrcoef(y_true, preds), 4)
                })
            
            # Save results
            res_df = pd.DataFrame(results)
            res_df = res_df[["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC Score"]]
            res_df.to_csv(test_csv, index=False)
            
            st.success("Test evaluation completed!")
            st.rerun()
            
        except KeyError as e:
            st.error(f"Missing column in test data: {e}")
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

if st.button("Evaluate on Test Data"):
    evaluate_test_set()

if os.path.exists(train_csv):
    st.subheader("Validation Set Results (Using valid.csv)")
    valid_results = pd.read_csv(train_csv)
    st.dataframe(valid_results, width='stretch', hide_index=True)
    
    import time
    mtime = os.path.getmtime(train_csv)
    st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")

    if has_models and os.path.exists('dataset/valid.csv'):
        st.subheader("Confusion Matrices (Validation Data)")

        try:
            valid_df = pd.read_csv('dataset/valid.csv')
            target = 'disease_diagnosis'

            if target in valid_df.columns:
                feat_cols = joblib.load('model/feature_columns.pkl')
                scaler = joblib.load('model/scaler.pkl')
                le = joblib.load('model/label_encoder.pkl')

                # Preprocess validation data (match training pipeline)
                y_true_val = le.transform(valid_df[target])
                X_val = valid_df.drop(columns=[target], errors='ignore')

                nums_val = X_val.select_dtypes(include=[np.number]).columns
                cats_val = X_val.select_dtypes(include=['object']).columns
                X_val[nums_val] = X_val[nums_val].fillna(X_val[nums_val].mean())
                X_val[cats_val] = X_val[cats_val].fillna('Unknown')

                X_val = pd.get_dummies(X_val, drop_first=True)
                X_val = X_val.reindex(columns=feat_cols, fill_value=0)
                X_val = scaler.transform(X_val)

                # Load all trained models
                model_files_val = [
                    f for f in os.listdir('model')
                    if f.endswith('.pkl') and f not in ['scaler.pkl', 'feature_columns.pkl', 'label_encoder.pkl']
                ]
                model_names_val = [f.replace('.pkl', '').replace('_', ' ').title() for f in model_files_val]

                tabs_val = st.tabs(model_names_val)

                for tab, mfile, mname in zip(tabs_val, model_files_val, model_names_val):
                    with tab:
                        mpath = os.path.join('model', mfile)
                        mdl = joblib.load(mpath)

                        y_pred_val = mdl.predict(X_val)
                        cm_val = confusion_matrix(y_true_val, y_pred_val)

                        st.write(f"**{mname} – Confusion Matrix (Validation)**")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm_val,
                            annot=True,
                            fmt='d',
                            cmap='Blues',
                            xticklabels=le.classes_,
                            yticklabels=le.classes_,
                            ax=ax
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        st.pyplot(fig)
            else:
                st.warning("Cannot compute confusion matrices: 'disease_diagnosis' column missing in validation data.")
        except Exception as e:
            st.error(f"Failed to compute validation confusion matrices: {e}")
else:
    st.info("Model comparison table not found. Train models to generate it.")

if os.path.exists(test_csv):
    st.subheader("Test Data Results")
    test_results = pd.read_csv(test_csv)
    st.dataframe(test_results, width='stretch', hide_index=True)
    
    import time
    mtime = os.path.getmtime(test_csv)
    st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")

if os.path.exists(test_csv) and has_models and os.path.exists('dataset/test.csv'):
    st.subheader("Confusion Matrices (Test Data)")

    try:
        test = pd.read_csv('dataset/test.csv')
        target = 'disease_diagnosis'

        if target in test.columns:
            feat_cols = joblib.load('model/feature_columns.pkl')
            scaler = joblib.load('model/scaler.pkl')
            le = joblib.load('model/label_encoder.pkl')

            # Preprocess test data (match training pipeline)
            y_true = le.transform(test[target])
            X = test.drop(columns=[target], errors='ignore')

            nums = X.select_dtypes(include=[np.number]).columns
            cats = X.select_dtypes(include=['object']).columns
            X[nums] = X[nums].fillna(X[nums].mean())
            X[cats] = X[cats].fillna('Unknown')

            X = pd.get_dummies(X, drop_first=True)
            X = X.reindex(columns=feat_cols, fill_value=0)
            X = scaler.transform(X)

            # Load all trained models
            model_files = [
                f for f in os.listdir('model')
                if f.endswith('.pkl') and f not in ['scaler.pkl', 'feature_columns.pkl', 'label_encoder.pkl']
            ]
            model_names = [f.replace('.pkl', '').replace('_', ' ').title() for f in model_files]

            tabs = st.tabs(model_names)

            for tab, mfile, mname in zip(tabs, model_files, model_names):
                with tab:
                    mpath = os.path.join('model', mfile)
                    mdl = joblib.load(mpath)

                    y_pred = mdl.predict(X)
                    cm = confusion_matrix(y_true, y_pred)

                    st.write(f"**{mname} – Confusion Matrix**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        xticklabels=le.classes_,
                        yticklabels=le.classes_,
                        ax=ax
                    )
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)
        else:
            st.warning("Cannot compute confusion matrices: 'disease_diagnosis' column missing in test data.")
    except Exception as e:
        st.error(f"Failed to compute confusion matrices: {e}")

# Uploaded file prediction
if has_models:
    st.header("4. Model Prediction")
    
    feat_cols = joblib.load('model/feature_columns.pkl')
    scaler = joblib.load('model/scaler.pkl')
    le = joblib.load('model/label_encoder.pkl')
    
    # Model selection in sidebar (shared across evaluation and batch prediction)
    mfiles = [f for f in os.listdir('model') if f.endswith('.pkl') and f not in ['scaler.pkl', 'feature_columns.pkl', 'label_encoder.pkl']]
    mnames = [f.replace('.pkl', '').replace('_', ' ').title() for f in mfiles]
    selected = st.sidebar.selectbox("Select Model", mnames, key="global_selected_model")
    
    mpath = os.path.join('model', f"{selected.lower().replace(' ', '_')}.pkl")
    mdl = joblib.load(mpath)
    
    st.subheader("Upload Test Data (CSV)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    # Optionally use the existing prepared test set
    if "use_existing_test" not in st.session_state:
        st.session_state.use_existing_test = False

    if os.path.exists('dataset/test.csv'):
        # Visual button that toggles a hidden checkbox-like flag
        btn_label = (
            "✓ Using existing prepared test data"
            if st.session_state.use_existing_test
            else "Use existing prepared test data"
        )
        if st.button(btn_label):
            st.session_state.use_existing_test = not st.session_state.use_existing_test
    else:
        st.caption("Prepared test dataset not found; only uploaded CSV can be used here.")

    data = None
    source_label = None

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        source_label = "Uploaded CSV"
    elif st.session_state.use_existing_test and os.path.exists('dataset/test.csv'):
        data = pd.read_csv('dataset/test.csv')
        source_label = "dataset/test.csv"

    if data is not None:
        st.write(f"Preview ({source_label}):", data.head())
        
        target = 'disease_diagnosis'
        
        if target not in data.columns:
            st.warning(f"Dataset must contain '{target}' column for evaluation.")
        else:
            if st.button("Predict & Evaluate"):
                # Preprocessing pipeline (matches training)
                y_true = le.transform(data[target])
                X = data.drop(columns=[target], errors='ignore')
                
                # Handle missing values
                nums = X.select_dtypes(include=[np.number]).columns
                cats = X.select_dtypes(include=['object']).columns
                X[nums] = X[nums].fillna(X[nums].mean())
                X[cats] = X[cats].fillna('Unknown')
                
                # One-hot encode and align columns
                X = pd.get_dummies(X, drop_first=True)
                X = X.reindex(columns=feat_cols, fill_value=0)
                X = scaler.transform(X)
                
                # Predict
                y_pred = mdl.predict(X)
                y_proba = mdl.predict_proba(X) if hasattr(mdl, "predict_proba") else None
                
                # Metrics
                st.subheader("Evaluation Metrics")
                c1, c2 = st.columns(2)
                
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                mcc = matthews_corrcoef(y_true, y_pred)
                
                auc = 0.0
                if y_proba is not None:
                    try:
                        auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
                    except ValueError:
                        pass
                    
                c1.metric("Accuracy", f"{acc:.4f}")
                c1.metric("Precision", f"{prec:.4f}")
                c1.metric("Recall", f"{rec:.4f}")
                
                c2.metric("F1 Score", f"{f1:.4f}")
                c2.metric("AUC Score", f"{auc:.4f}")
                c2.metric("MCC Score", f"{mcc:.4f}")
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=le.classes_, yticklabels=le.classes_)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)

# Single patient prediction
if has_models:
    st.header("5. Single Instance Prediction")
    st.write("Enter patient information to get predictions from all models:")
    
    # Load defaults from JSON
    import json
    defaults = {}
    if os.path.exists('dataset/default_values.json'):
        with open('dataset/default_values.json', 'r') as f:
            defaults = json.load(f)
    
    def get_def(attr, fallback):
        return defaults.get(attr, {}).get('default', fallback) if attr in defaults else fallback
    
    def get_rng(attr, fallback_min, fallback_max):
        if attr in defaults:
            return defaults[attr].get('min', fallback_min), defaults[attr].get('max', fallback_max)
        return fallback_min, fallback_max
    
    def get_opts(attr, fallback_opts):
        if attr in defaults and 'options' in defaults[attr]:
            opts = defaults[attr]['options']
            # Remove NaN values
            clean = [o for o in opts if (isinstance(o, (int, float)) and pd.notna(o)) or 
                                        (isinstance(o, str) and o.lower() != 'nan') or
                                        (o is not None and str(o).lower() != 'nan')]
            return clean if clean else fallback_opts
        return fallback_opts
    
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age_min, age_max = get_rng('age', 0, 120)
            age = st.number_input("Age", min_value=int(age_min), max_value=int(age_max), 
                                 value=int(get_def('age', 51)), step=1)
            
            gender_opts = get_opts('gender', ["Male", "Female"])
            gender_def = get_def('gender', 'Male')
            gender = st.selectbox("Gender", gender_opts, 
                                 index=gender_opts.index(gender_def) if gender_def in gender_opts else 0)
            
            bmi_min, bmi_max = get_rng('bmi', 10.0, 50.0)
            bmi = st.number_input("BMI", min_value=float(bmi_min), max_value=float(bmi_max), 
                                 value=float(get_def('bmi', 26.2)), step=0.1)
            
            smoke_opts = get_opts('smoking_status', ["Never", "Former", "Current"])
            smoke_def = get_def('smoking_status', 'Never')
            smoking_status = st.selectbox("Smoking Status", smoke_opts,
                                        index=smoke_opts.index(smoke_def) if smoke_def in smoke_opts else 0)
            
            alc_opts = get_opts('alcohol_consumption', ["None", "Moderate", "Heavy"])
            alc_opts = ["None" if (pd.isna(o) or str(o) == 'nan') else o for o in alc_opts]
            if "None" not in alc_opts:
                alc_opts.insert(0, "None")
            alc_def = get_def('alcohol_consumption', 'Moderate')
            if pd.isna(alc_def) or str(alc_def) == 'nan':
                alc_def = "None"
            alcohol_consumption = st.selectbox("Alcohol Consumption", alc_opts,
                                              index=alc_opts.index(alc_def) if alc_def in alc_opts else 0)
            
            ex_opts = get_opts('exercise_level', ["Sedentary", "Light", "Moderate", "Active"])
            ex_def = get_def('exercise_level', 'Sedentary')
            exercise_level = st.selectbox("Exercise Level", ex_opts,
                                         index=ex_opts.index(ex_def) if ex_def in ex_opts else 0)
        
        with c2:
            diet_opts = get_opts('diet_type', ["Omnivore", "Vegetarian", "Pescatarian", "Vegan"])
            diet_type = st.selectbox("Diet Type", diet_opts,
                                     index=diet_opts.index(get_def('diet_type', 'Vegan')) if get_def('diet_type', 'Vegan') in diet_opts else 0)
            sun_opts = get_opts('sun_exposure', ["Low", "Moderate", "High"])
            sun_exposure = st.selectbox("Sun Exposure", sun_opts,
                                       index=sun_opts.index(get_def('sun_exposure', 'Moderate')) if get_def('sun_exposure', 'Moderate') in sun_opts else 0)
            inc_opts = get_opts('income_level', ["Low", "Middle", "High"])
            income_level = st.selectbox("Income Level", inc_opts,
                                       index=inc_opts.index(get_def('income_level', 'High')) if get_def('income_level', 'High') in inc_opts else 0)
            lat_opts = get_opts('latitude_region', ["Low", "Mid", "High"])
            latitude_region = st.selectbox("Latitude Region", lat_opts,
                                          index=lat_opts.index(get_def('latitude_region', 'Low')) if get_def('latitude_region', 'Low') in lat_opts else 0)
            
            va_min, va_max = get_rng('vitamin_a_percent_rda', 0.0, 500.0)
            vitamin_a_percent_rda = st.number_input("Vitamin A (% RDA)", 
                                                    min_value=float(va_min), max_value=float(va_max), 
                                                    value=float(get_def('vitamin_a_percent_rda', 85.5)), step=0.1)
            vc_min, vc_max = get_rng('vitamin_c_percent_rda', 0.0, 500.0)
            vitamin_c_percent_rda = st.number_input("Vitamin C (% RDA)", 
                                                    min_value=float(vc_min), max_value=float(vc_max), 
                                                    value=float(get_def('vitamin_c_percent_rda', 83.5)), step=0.1)
        
        with c3:
            vd_min, vd_max = get_rng('vitamin_d_percent_rda', 0.0, 500.0)
            vitamin_d_percent_rda = st.number_input("Vitamin D (% RDA)", 
                                                    min_value=float(vd_min), max_value=float(vd_max), 
                                                    value=float(get_def('vitamin_d_percent_rda', 62.27)), step=0.1)
            ve_min, ve_max = get_rng('vitamin_e_percent_rda', 0.0, 500.0)
            vitamin_e_percent_rda = st.number_input("Vitamin E (% RDA)", 
                                                    min_value=float(ve_min), max_value=float(ve_max), 
                                                    value=float(get_def('vitamin_e_percent_rda', 84.05)), step=0.1)
            vb12_min, vb12_max = get_rng('vitamin_b12_percent_rda', 0.0, 500.0)
            vitamin_b12_percent_rda = st.number_input("Vitamin B12 (% RDA)", 
                                                       min_value=float(vb12_min), max_value=float(vb12_max), 
                                                       value=float(get_def('vitamin_b12_percent_rda', 55.6)), step=0.1)
            fol_min, fol_max = get_rng('folate_percent_rda', 0.0, 500.0)
            folate_percent_rda = st.number_input("Folate (% RDA)", 
                                                 min_value=float(fol_min), max_value=float(fol_max), 
                                                 value=float(get_def('folate_percent_rda', 84.8)), step=0.1)
            ca_min, ca_max = get_rng('calcium_percent_rda', 0.0, 500.0)
            calcium_percent_rda = st.number_input("Calcium (% RDA)", 
                                                  min_value=float(ca_min), max_value=float(ca_max), 
                                                  value=float(get_def('calcium_percent_rda', 77.1)), step=0.1)
            fe_min, fe_max = get_rng('iron_percent_rda', 0.0, 500.0)
            iron_percent_rda = st.number_input("Iron (% RDA)", 
                                              min_value=float(fe_min), max_value=float(fe_max), 
                                              value=float(get_def('iron_percent_rda', 71.25)), step=0.1)
        
        st.subheader("Clinical Measurements")
        c4, c5, c6 = st.columns(3)
        
        with c4:
            hb_min, hb_max = get_rng('hemoglobin_g_dl', 5.0, 20.0)
            hemoglobin_g_dl = st.number_input("Hemoglobin (g/dL)", 
                                              min_value=float(hb_min), max_value=float(hb_max), 
                                              value=float(get_def('hemoglobin_g_dl', 14.1)), step=0.1)
            svd_min, svd_max = get_rng('serum_vitamin_d_ng_ml', 0.0, 100.0)
            serum_vitamin_d_ng_ml = st.number_input("Serum Vitamin D (ng/mL)", 
                                                    min_value=float(svd_min), max_value=float(svd_max), 
                                                    value=float(get_def('serum_vitamin_d_ng_ml', 18.4)), step=0.1)
        
        with c5:
            svb12_min, svb12_max = get_rng('serum_vitamin_b12_pg_ml', 0.0, 1500.0)
            serum_vitamin_b12_pg_ml = st.number_input("Serum Vitamin B12 (pg/mL)", 
                                                      min_value=float(svb12_min), max_value=float(svb12_max), 
                                                      value=float(get_def('serum_vitamin_b12_pg_ml', 214.85)), step=0.1)
            svf_min, svf_max = get_rng('serum_folate_ng_ml', 0.0, 50.0)
            serum_folate_ng_ml = st.number_input("Serum Folate (ng/mL)", 
                                                 min_value=float(svf_min), max_value=float(svf_max), 
                                                 value=float(get_def('serum_folate_ng_ml', 10.0)), step=0.1)
        
        with c6:
            # Symptoms count is categorical (fake numeric)
            sym_opts = get_opts('symptoms_count', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            sym_opts = [int(float(x)) for x in sym_opts]
            sym_def = int(float(get_def('symptoms_count', 0)))
            symptoms_count = st.selectbox("Symptoms Count", sym_opts,
                                         index=sym_opts.index(sym_def) if sym_def in sym_opts else 0)
        
        st.subheader("Symptoms")
        c7, c8 = st.columns(2)
        
        with c7:
            has_night_blindness = st.checkbox("Night Blindness")
            has_fatigue = st.checkbox("Fatigue")
            has_bleeding_gums = st.checkbox("Bleeding Gums")
            has_bone_pain = st.checkbox("Bone Pain")
            has_muscle_weakness = st.checkbox("Muscle Weakness")
        
        with c8:
            has_numbness_tingling = st.checkbox("Numbness/Tingling")
            has_memory_problems = st.checkbox("Memory Problems")
            has_pale_skin = st.checkbox("Pale Skin")
            has_multiple_deficiencies = st.checkbox("Multiple Deficiencies")
        
        submitted = st.form_submit_button("Predict", use_container_width=True)
        
        if submitted:
            try:
                # Load preprocessing artifacts
                feat_cols = joblib.load('model/feature_columns.pkl')
                scaler = joblib.load('model/scaler.pkl')
                le = joblib.load('model/label_encoder.pkl')
                
                # Convert "None" to None for alcohol to match dataset format
                alc_val = None if alcohol_consumption == "None" else alcohol_consumption
                
                patient_data = {
                    'age': age, 'gender': gender, 'bmi': bmi,
                    'smoking_status': smoking_status, 'alcohol_consumption': alc_val,
                    'exercise_level': exercise_level, 'diet_type': diet_type,
                    'sun_exposure': sun_exposure, 'income_level': income_level,
                    'latitude_region': latitude_region,
                    'vitamin_a_percent_rda': vitamin_a_percent_rda,
                    'vitamin_c_percent_rda': vitamin_c_percent_rda,
                    'vitamin_d_percent_rda': vitamin_d_percent_rda,
                    'vitamin_e_percent_rda': vitamin_e_percent_rda,
                    'vitamin_b12_percent_rda': vitamin_b12_percent_rda,
                    'folate_percent_rda': folate_percent_rda,
                    'calcium_percent_rda': calcium_percent_rda,
                    'iron_percent_rda': iron_percent_rda,
                    'hemoglobin_g_dl': hemoglobin_g_dl,
                    'serum_vitamin_d_ng_ml': serum_vitamin_d_ng_ml,
                    'serum_vitamin_b12_pg_ml': serum_vitamin_b12_pg_ml,
                    'serum_folate_ng_ml': serum_folate_ng_ml,
                    'symptoms_count': symptoms_count,
                    'has_night_blindness': 1 if has_night_blindness else 0,
                    'has_fatigue': 1 if has_fatigue else 0,
                    'has_bleeding_gums': 1 if has_bleeding_gums else 0,
                    'has_bone_pain': 1 if has_bone_pain else 0,
                    'has_muscle_weakness': 1 if has_muscle_weakness else 0,
                    'has_numbness_tingling': 1 if has_numbness_tingling else 0,
                    'has_memory_problems': 1 if has_memory_problems else 0,
                    'has_pale_skin': 1 if has_pale_skin else 0,
                    'has_multiple_deficiencies': 1 if has_multiple_deficiencies else 0
                }
                
                X = pd.DataFrame([patient_data])
                
                # Preprocess (match training pipeline)
                nums = X.select_dtypes(include=[np.number]).columns
                cats = X.select_dtypes(include=['object']).columns
                X[nums] = X[nums].fillna(X[nums].mean())
                X[cats] = X[cats].fillna('Unknown')
                
                X = pd.get_dummies(X, drop_first=True)
                X = X.reindex(columns=feat_cols, fill_value=0)
                X = scaler.transform(X)
                
                # Get predictions from all models
                mfiles = [f for f in os.listdir('model') 
                         if f.endswith('.pkl') and f not in ['scaler.pkl', 'feature_columns.pkl', 'label_encoder.pkl']]
                preds = []
                
                for mfile in mfiles:
                    mname = mfile.replace('.pkl', '').replace('_', ' ').title()
                    mpath = os.path.join('model', mfile)
                    mdl = joblib.load(mpath)
                    
                    pred = mdl.predict(X)[0]
                    pred_class = le.inverse_transform([pred])[0]
                    
                    if hasattr(mdl, "predict_proba"):
                        proba = mdl.predict_proba(X)[0]
                        max_prob = np.max(proba)
                        prob_dict = {le.classes_[i]: round(proba[i], 4) for i in range(len(proba))}
                    else:
                        max_prob = None
                        prob_dict = None
                    
                    preds.append({
                        "Model": mname,
                        "Prediction": pred_class,
                        "Confidence": f"{max_prob:.4f}" if max_prob else "N/A",
                        "Probabilities": prob_dict
                    })
                
                # Display results
                st.subheader("Prediction Results")
                pred_df = pd.DataFrame(preds)
                st.dataframe(pred_df[["Model", "Prediction", "Confidence"]], width='stretch', hide_index=True)
                
                # Detailed probabilities
                st.subheader("Detailed Probabilities")
                for p in preds:
                    with st.expander(f"{p['Model']} - {p['Prediction']}"):
                        if p['Probabilities']:
                            prob_df = pd.DataFrame(list(p['Probabilities'].items()), 
                                                   columns=['Disease', 'Probability'])
                            prob_df = prob_df.sort_values('Probability', ascending=False)
                            st.dataframe(prob_df, width='stretch', hide_index=True)
                        else:
                            st.info("Probability information not available for this model.")
                
            except KeyError as e:
                st.error(f"Missing required field: {e}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Multiple-instance prediction (CSV without target column)
if has_models:
    st.header("6. Multiple Instance Prediction (Unlabeled CSV)")
    st.write(
        "Upload a CSV without the target column to generate predictions for multiple patients. "
        "You can also choose to use the existing prepared test data (without labels). "
        "Predictions will be generated using the model selected in the sidebar."
    )

    # Reuse sidebar-selected model from section 4 (no new widget)
    feat_cols = joblib.load('model/feature_columns.pkl')
    scaler = joblib.load('model/scaler.pkl')
    le = joblib.load('model/label_encoder.pkl')

    mfiles = [f for f in os.listdir('model') if f.endswith('.pkl') and f not in ['scaler.pkl', 'feature_columns.pkl', 'label_encoder.pkl']]
    mnames = [f.replace('.pkl', '').replace('_', ' ').title() for f in mfiles]

    # Use the same selection as section 4; do NOT create a second widget
    if 'selected' in locals() and selected in mnames:
        selected_multi = selected
    else:
        selected_multi = mnames[0] if mnames else None

    mpath_multi = os.path.join('model', f"{selected_multi.lower().replace(' ', '_')}.pkl")
    mdl_multi = joblib.load(mpath_multi)

    st.subheader("Upload Unlabeled Data (CSV)")
    uploaded_multi = st.file_uploader("Upload CSV without target column", type=["csv"], key="multi_csv_uploader")

    # Toggle for using existing prepared test set (features only)
    if "use_existing_unlabeled_test" not in st.session_state:
        st.session_state.use_existing_unlabeled_test = False

    if os.path.exists('dataset/test.csv'):
        btn_label_multi = (
            "✓ Using existing prepared test data (ignoring labels)"
            if st.session_state.use_existing_unlabeled_test
            else "Use existing prepared test data (ignoring labels)"
        )
        if st.button(btn_label_multi, key="multi_use_existing"):
            st.session_state.use_existing_unlabeled_test = not st.session_state.use_existing_unlabeled_test
    else:
        st.caption("Prepared test dataset not found; only uploaded CSV can be used here.")

    data_multi = None
    source_multi = None

    if uploaded_multi is not None:
        data_multi = pd.read_csv(uploaded_multi)
        source_multi = "Uploaded CSV"
    elif st.session_state.use_existing_unlabeled_test and os.path.exists('dataset/test.csv'):
        data_multi = pd.read_csv('dataset/test.csv')
        # If test.csv includes labels, drop them to simulate unlabeled prediction
        if 'disease_diagnosis' in data_multi.columns:
            data_multi = data_multi.drop(columns=['disease_diagnosis'])
        source_multi = "dataset/test.csv (features only)"

    if data_multi is not None:
        st.write(f"Preview ({source_multi}):", data_multi.head())

        if 'disease_diagnosis' in data_multi.columns:
            st.warning("The input should not contain the 'disease_diagnosis' column. It will be ignored for prediction.")
            data_multi = data_multi.drop(columns=['disease_diagnosis'])

        if st.button("Generate Predictions", key="multi_predict"):
            try:
                X_multi = data_multi.copy()

                # Handle missing values
                nums_m = X_multi.select_dtypes(include=[np.number]).columns
                cats_m = X_multi.select_dtypes(include=['object']).columns
                X_multi[nums_m] = X_multi[nums_m].fillna(X_multi[nums_m].mean())
                X_multi[cats_m] = X_multi[cats_m].fillna('Unknown')

                # One-hot encode and align columns
                X_multi = pd.get_dummies(X_multi, drop_first=True)
                X_multi = X_multi.reindex(columns=feat_cols, fill_value=0)
                X_multi = scaler.transform(X_multi)

                # Predict
                y_pred_multi = mdl_multi.predict(X_multi)
                y_pred_labels = le.inverse_transform(y_pred_multi)

                results_df = data_multi.copy()
                results_df["Predicted_Diagnosis"] = y_pred_labels

                if hasattr(mdl_multi, "predict_proba"):
                    proba_multi = mdl_multi.predict_proba(X_multi)
                    max_proba = proba_multi.max(axis=1)
                    results_df["Prediction_Confidence"] = max_proba

                # Reorder columns so Predicted_Diagnosis is last
                base_cols = [
                    c
                    for c in results_df.columns
                    if c not in ["Predicted_Diagnosis", "Prediction_Confidence"]
                ]
                ordered_cols = base_cols
                if "Prediction_Confidence" in results_df.columns:
                    ordered_cols.append("Prediction_Confidence")
                ordered_cols.append("Predicted_Diagnosis")
                results_df = results_df[ordered_cols]

                st.subheader("Prediction Results")
                st.dataframe(
                    results_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Predicted_Diagnosis": st.column_config.Column(
                            "Predicted_Diagnosis", pinned=True
                        )
                    },
                )

                # Offer download of predictions
                csv_out = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_out,
                    file_name="predictions_unlabeled.csv",
                    mime="text/csv",
                    key="multi_download"
                )

            except Exception as e:
                st.error(f"Multiple-instance prediction failed: {e}")
else:
    st.info("Please train models first to enable prediction functionality.")
