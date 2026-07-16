import streamlit as st

st.set_page_config(page_title="Dokumentasi", page_icon="📖", layout="wide")
st.title("📖 Dokumentasi Proyek")

st.markdown(
    """
## Tentang Dataset

Dataset yang digunakan adalah **Web Page Phishing Detection Dataset** yang dibangun oleh
Hannousse & Yahiouche (2021) dalam studi *"Towards benchmark datasets for machine learning based
website phishing detection: An experimental study"*, dipublikasikan di Mendeley Data.

- **Jumlah sampel**: 11.430 URL
- **Jumlah fitur**: 87 fitur (56 dari struktur/sintaks URL, 24 dari konten halaman, 7 dari layanan eksternal)
- **Target**: `status` (legitimate / phishing), seimbang 50:50
- **Sumber**: https://data.mendeley.com/datasets/c2gw7fy2j4/3

## Metodologi

Pipeline Machine Learning yang digunakan mengikuti alur standar:

1. **Problem Definition & Data Acquisition** — menentukan masalah klasifikasi biner deteksi phishing dan mengumpulkan dataset.
2. **Exploratory Data Analysis & Preprocessing** — analisis kualitas data, distribusi kelas, korelasi fitur, encoding target, feature scaling, train-test split (80:20, stratified).
3. **Modeling & Evaluation** — melatih dan membandingkan 2 algoritma (Decision Tree & Random Forest), hyperparameter tuning dengan GridSearchCV (5-fold CV), evaluasi menggunakan Accuracy, Precision, Recall, F1-Score, AUC-ROC, serta interpretasi model dengan SHAP.
4. **Deployment** — aplikasi Streamlit ini, terdiri dari Dashboard EDA, Model Demo, Evaluasi Model, dan Interpretasi Hasil.

## Model Terbaik

**Random Forest (Tuned)** dipilih sebagai model final berdasarkan performa Accuracy, F1-Score, dan
AUC-ROC yang lebih tinggi dan lebih stabil dibandingkan Decision Tree, serta ketahanannya terhadap
overfitting sebagai model ensemble.

## Cara Penggunaan Aplikasi

1. Buka halaman **📊 Dashboard EDA** untuk menjelajahi karakteristik dataset secara interaktif.
2. Buka halaman **🔍 Model Demo** untuk mencoba prediksi status website dengan mengisi nilai fitur utama secara manual.
3. Buka halaman **📈 Evaluasi Model** untuk melihat detail performa model (confusion matrix, ROC curve, classification report).
4. Buka halaman **🧠 Interpretasi Hasil** untuk memahami fitur-fitur yang paling berpengaruh terhadap keputusan model.

## Struktur Repository

```
capstone-project-phishing-detection/
├── data/
│   ├── raw/                  # Data mentah (dataset_phishing.csv)
│   └── processed/            # Hasil pipeline (results.json, model_comparison.csv)
├── notebooks/
│   └── 01_eda_modeling.ipynb # EDA, preprocessing, modeling, evaluasi
├── src/
│   └── run_pipeline.py       # Script pipeline end-to-end
├── models/
│   ├── best_random_forest.pkl
│   ├── best_decision_tree.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── app/
│   ├── app.py                 # Entry point Streamlit
│   └── pages/                 # Halaman-halaman aplikasi
├── reports/
│   └── Laporan_Teknis.pdf
├── requirements.txt
└── README.md
```

## Referensi

- Hannousse, A. & Yahiouche, S. (2021). *Towards benchmark datasets for machine learning based website
  phishing detection: An experimental study*. Engineering Applications of Artificial Intelligence.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.
- Lundberg, S. & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP).
- Streamlit Documentation — https://docs.streamlit.io
"""
)
