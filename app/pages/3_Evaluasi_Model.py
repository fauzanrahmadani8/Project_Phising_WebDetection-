import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report

st.set_page_config(page_title="Evaluasi Model", page_icon="📈", layout="wide")
st.title("📈 Evaluasi Model")

BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "..", "..", "data", "raw", "dataset_phishing.csv")
COMPARISON_PATH = os.path.join(BASE, "..", "..", "data", "processed", "model_comparison.csv")
RF_MODEL_PATH = os.path.join(BASE, "..", "..", "models", "best_random_forest.pkl")
DT_MODEL_PATH = os.path.join(BASE, "..", "..", "models", "best_decision_tree.pkl")
FEATCOLS_PATH = os.path.join(BASE, "..", "..", "models", "feature_columns.pkl")


@st.cache_data
def load_test_split():
    df = pd.read_csv(DATA_PATH)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["status_encoded"] = le.fit_transform(df["status"])
    df_model = df.drop(columns=["url", "status"])
    X = df_model.drop(columns=["status_encoded"])
    y = df_model["status_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_test, y_test


try:
    comparison = pd.read_csv(COMPARISON_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    dt_model = joblib.load(DT_MODEL_PATH)
    feature_cols = joblib.load(FEATCOLS_PATH)
    X_test, y_test = load_test_split()
except FileNotFoundError:
    st.error("Artefak model/hasil belum ditemukan. Jalankan notebook 01_eda_modeling.ipynb terlebih dahulu.")
    st.stop()

st.subheader("Tabel Perbandingan Performa Semua Model")
st.dataframe(comparison.style.highlight_max(axis=0, subset=comparison.columns[1:], color="#c6e2c6"), use_container_width=True)

st.markdown("---")

model_choice = st.radio("Pilih model untuk detail evaluasi:", ["Random Forest (Tuned)", "Decision Tree (Tuned)"], horizontal=True)
model = rf_model if "Random Forest" in model_choice else dt_model

X_test_ordered = X_test[feature_cols]
y_pred = model.predict(X_test_ordered)
y_prob = model.predict_proba(X_test_ordered)[:, 1]

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"#### Confusion Matrix — {model_choice}")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    cmap = "Blues" if "Random Forest" in model_choice else "Greens"
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    st.pyplot(fig)

with col2:
    st.markdown(f"#### Kurva ROC — {model_choice}")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="#2E86AB")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

st.markdown("#### Classification Report")
report = classification_report(y_test, y_pred, target_names=["legitimate", "phishing"], output_dict=True)
st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

st.markdown("---")
st.markdown("### 🏆 Kesimpulan Pemilihan Model")
st.success(
    "**Random Forest (Tuned)** dipilih sebagai model terbaik karena memiliki kombinasi Accuracy, "
    "Precision, Recall, F1-Score, dan AUC-ROC yang lebih tinggi dan lebih stabil dibandingkan Decision Tree, "
    "serta lebih tahan overfitting sebagai model ensemble."
)
